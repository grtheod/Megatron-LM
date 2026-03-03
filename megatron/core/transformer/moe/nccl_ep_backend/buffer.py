# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""NCCLEPBuffer: High-level Python interface for NCCL EP dispatch and combine operations.

This module provides a Buffer class similar to DeepEP's Buffer, adapted for Megatron-LM's
token dispatcher framework. It wraps the low-level NCCL EP C API with a Pythonic interface.

REQUIREMENTS:
    1. NCCL library with MoE extensions
    2. PyTorch distributed initialized

Example setup:
    export NCCL_HOME=/path/to/nccl/build
    export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH}"
    torchrun --nproc_per_node=8 your_training_script.py
"""

import ctypes
import ctypes.util
import logging
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist

try:
    from nccl_ep import (
        NCCLLibrary,
        get_nccl_comm_from_group,
        ncclDataTypeEnum,
        ncclEpAlgorithm_t,
        ncclEpDispatchConfig_t,
        ncclEpGroupConfig_t,
        ncclEpTensorTag_t,
        ncclNDTensor_t,
        ncclEpAllocFn_t,
        ncclEpFreeFn_t,
        CUDA_SUCCESS,
        CUDA_ERROR_MEMORY_ALLOCATION,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import NCCL EP wrapper. "
        "Please install the nccl-ep package from: "
        "path/to/nccl/nccl-ep/python\n"
        f"Error: {e}"
    )

# Logger for NCCL EP buffer operations
logger = logging.getLogger(__name__)

# Global tensor map to prevent garbage collection (PyTorch allocator integration)
_tensor_map: dict = {}

# Global references to prevent garbage collection of callback wrappers
_pytorch_alloc_fn = None
_pytorch_free_fn = None

# Cached CUDA runtime library for cudaHostAlloc/cudaFreeHost
_cudart = None


def _get_cudart():
    """Load and cache the CUDA runtime library. Raises RuntimeError if not found."""
    global _cudart
    if _cudart is None:
        cuda_lib_name = ctypes.util.find_library('cudart')
        if cuda_lib_name is not None:
            _cudart = ctypes.CDLL(cuda_lib_name)
        else:
            for lib_name in ['libcudart.so', 'libcudart.so.12', 'libcudart.so.11']:
                try:
                    _cudart = ctypes.CDLL(lib_name)
                    break
                except OSError:
                    continue
            else:
                raise RuntimeError("Could not find CUDA runtime library (libcudart.so)")
    return _cudart


def _pytorch_alloc_callback(ptr_ptr, size: int) -> int:
    """Allocate memory from PyTorch's caching allocator for NCCL EP.

    Uses PyTorch's memory pool instead of cudaMalloc for efficiency.

    Args:
        ptr_ptr: Pointer to pointer where allocated address will be stored
        size: Number of bytes to allocate

    Returns:
        cudaError_t (0 = cudaSuccess, 2 = cudaErrorMemoryAllocation)
    """
    try:
        # Allocate a byte tensor of the requested size
        tensor = torch.empty(size, dtype=torch.uint8, device='cuda')
        ptr = tensor.data_ptr()

        # Store tensor to prevent garbage collection
        _tensor_map[ptr] = tensor

        # Write pointer value to output parameter
        ptr_ptr[0] = ptr
        return CUDA_SUCCESS
    except Exception as e:
        logger.error(f"PyTorch alloc failed for {size} bytes: {e}")
        return CUDA_ERROR_MEMORY_ALLOCATION


def _pytorch_free_callback(ptr: int) -> int:
    """Free memory back to PyTorch's caching allocator.

    Args:
        ptr: Pointer to memory to free

    Returns:
        cudaError_t (0 = cudaSuccess)
    """
    try:
        # Remove from map, tensor will be garbage collected
        _tensor_map.pop(ptr, None)
        return CUDA_SUCCESS
    except Exception as e:
        logger.error(f"PyTorch free failed: {e}")
        return CUDA_SUCCESS  # Don't fail on free errors


def _get_pytorch_allocator_callbacks():
    """Get or create PyTorch allocator callback wrappers.

    Returns:
        Tuple of (alloc_callback, free_callback) ctypes function pointers
    """
    global _pytorch_alloc_fn, _pytorch_free_fn

    if _pytorch_alloc_fn is None:
        # Ensure CUDA is initialized
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        torch.cuda.init()
        _ = torch.empty(1, device="cuda")  # Force context creation

        # Create callback wrappers (must keep references to prevent GC)
        _pytorch_alloc_fn = ncclEpAllocFn_t(_pytorch_alloc_callback)
        _pytorch_free_fn = ncclEpFreeFn_t(_pytorch_free_callback)

    return _pytorch_alloc_fn, _pytorch_free_fn

# Control logging verbosity via environment variable
# Set NCCL_EP_LOG_LEVEL=INFO or NCCL_EP_LOG_LEVEL=DEBUG to see logs
_NCCL_EP_LOG_LEVEL = os.environ.get("NCCL_EP_LOG_LEVEL", "WARNING").upper()
if _NCCL_EP_LOG_LEVEL == "DEBUG":
    logger.setLevel(logging.DEBUG)
elif _NCCL_EP_LOG_LEVEL == "INFO":
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.WARNING)

# Add handler with minimal format to match NCCL output style (message only, no Python prefix)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_handler)
    logger.propagate = False  # Don't duplicate to root logger


def _nccl_log_prefix() -> str:
    """Generate NCCL-style log prefix: hostname:pid:pid [local_rank]"""
    import socket
    pid = os.getpid()
    hostname = socket.gethostname()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return f"{hostname}:{pid}:{pid} [{local_rank}]"


class NCCLEPEventOverlap:
    """Event-based async completion tracking.

    This class enables event chaining for overlapping communication with computation.
    It records a CUDA event on a stream and provides methods for synchronization.

    Usage:
        # After launching async operation
        event = NCCLEPEventOverlap(comm_stream)

        # Later, when result is needed
        event.current_stream_wait()  # GPU-side sync, no CPU blocking

        # Or for explicit CPU sync (use sparingly)
        event.synchronize()
    """
    __slots__ = ['event', 'stream']

    def __init__(self, stream: torch.cuda.Stream):
        """Create an event overlap tracker.

        Args:
            stream: The CUDA stream to record the event on
        """
        self.stream = stream
        self.event = torch.cuda.Event()
        self.event.record(stream)

    def current_stream_wait(self):
        """Make the current CUDA stream wait for this event (GPU-side, non-blocking)."""
        torch.cuda.current_stream().wait_event(self.event)

    def synchronize(self):
        """Block CPU until the event completes. Use sparingly - prefer current_stream_wait()."""
        self.event.synchronize()

    def query(self) -> bool:
        """Check if the event has completed without blocking."""
        return self.event.query()


class NDTensorWrapper:
    """Wrapper that encapsulates ncclNDTensor_t with its sizes/strides arrays.

    This class keeps the ctypes arrays alive as long as the wrapper exists,
    preventing dangling pointers when the nd_tensor struct references them.

    Usage:
        wrapper = NDTensorWrapper(tensor, tag=ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS)
        # Pass wrapper.nd_tensor to C API
        ctypes.pointer(wrapper.nd_tensor)

        # Later, update data pointer for reuse:
        wrapper.update_data_ptr(new_tensor)
    """
    __slots__ = ['nd_tensor', '_sizes', '_strides', '_pointer', '_max_ndim']

    def __init__(self, tensor: Optional[torch.Tensor] = None, tag: int = 0, flags: int = 0, max_ndim: int = 2):
        """Create an NDTensorWrapper from a PyTorch tensor or pre-allocate for later use.

        Args:
            tensor: PyTorch tensor (must be contiguous), or None to pre-allocate
            tag: Tensor tag (e.g., NCCL_EP_TENSOR_TAG_TOKENS)
            flags: Tensor flags
            max_ndim: Maximum number of dimensions (for pre-allocation)
        """
        self._max_ndim = max_ndim
        self.nd_tensor = ncclNDTensor_t()
        self.nd_tensor.version = 1

        # Pre-allocate arrays at max size
        self._sizes = (ctypes.c_uint * max_ndim)()
        self._strides = (ctypes.c_uint * max_ndim)()
        self.nd_tensor.sizes = ctypes.cast(self._sizes, ctypes.POINTER(ctypes.c_uint))
        self.nd_tensor.strides = ctypes.cast(self._strides, ctypes.POINTER(ctypes.c_uint))

        # Pre-create pointer
        self._pointer = ctypes.pointer(self.nd_tensor)

        if tensor is not None:
            self.configure(tensor, tag, flags)

    def configure(self, tensor: torch.Tensor, tag: int = 0, flags: int = 0):
        """Configure this wrapper for a tensor. Updates values in-place, no allocations."""
        ndim = len(tensor.shape)
        assert ndim <= self._max_ndim, f"ndim {ndim} exceeds max_ndim {self._max_ndim}"

        self.nd_tensor.ndim = ndim
        for i in range(ndim):
            self._sizes[i] = tensor.shape[i]
            self._strides[i] = 1

        self.nd_tensor.datatype = ncclDataTypeEnum.from_torch(tensor.dtype)
        self.nd_tensor.data = tensor.data_ptr()
        self.nd_tensor.tag = tag
        self.nd_tensor.flags = flags

    def update_data_ptr(self, tensor: torch.Tensor):
        """Update only the data pointer for reuse with a new tensor.

        Note: The new tensor must have the same shape and dtype as the original.
        """
        self.nd_tensor.data = tensor.data_ptr()

    def get_pointer(self):
        """Get a ctypes pointer to the nd_tensor for passing to C API."""
        return self._pointer


class NCCLEPBuffer:
    """Buffer for NCCL EP dispatch and combine operations.

    This class manages the lifecycle of NCCL EP groups and handles, providing
    high-level dispatch and combine operations for MoE token routing.

    Uses NCCL EP's high-throughput kernels. Runs in sync-only mode; async_finish and
    allocate_on_comm_stream are kept for API compatibility with the flex dispatcher
    but are ignored. Uses a dedicated communication stream (separate from compute stream).

    Handle Lifecycle:
        A handle is created once per forward pass and reused for the backward pass:
        1. Forward dispatch (creates handle with topk_idx)
        2. Forward combine (uses handle, no topk_weights needed)
        3. Backward: combine.backward -> dispatch (cached mode, handle only)
        4. Backward: dispatch.backward -> combine (with topk_weights for gradient)
        5. Handle destroyed after backward pass completes

    Example:
        buffer = NCCLEPBuffer(...)

        # Forward pass
        recv_x, recv_idx, recv_probs, counts, handle = buffer.dispatch(
            x, topk_idx, topk_weights
        )
        # ... expert computation ...
        combined, _ = buffer.combine(expert_out, handle)  # No topk_weights in forward

        # Backward pass (same handle, operations in reverse)
        # Step 1: combine.backward dispatches gradients using cached routing
        grad_expert, _, _, _, _ = buffer.dispatch(grad_combined, handle=handle)
        # Step 2: dispatch.backward combines gradients with weight gradients
        grad_x, grad_weights = buffer.combine(grad_recv, handle, topk_weights=grad_probs)

        # Clean up after backward
        buffer.destroy_handle(handle)
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_experts: int,
        hidden_dim: int,
        max_tokens_per_rank: int,
        algorithm: int = ncclEpAlgorithm_t.NCCL_EP_ALGO_HIGH_THROUGHPUT,
        rdma_buffer_size: int = 0,  # 0 means auto
        num_qp_per_rank: int = 20,  # Must satisfy: num_qp >= num_channels or num_qp >= num_sms
        num_channels: int = 10,  # num_sms = num_channels * 2
        use_pytorch_allocator: bool = True,  # Use PyTorch's caching allocator
    ):
        """Initialize NCCL EP buffer.

        Args:
            group: PyTorch distributed process group (must use NCCL backend)
            num_experts: Total number of experts across all ranks
            hidden_dim: Hidden dimension size
            max_tokens_per_rank: Max tokens sent per rank (C++ config.max_tokens_per_rank). Must be > 0 for HT; dynamic (0) not supported.
            algorithm: NCCL EP algorithm (LOW_LATENCY or HIGH_THROUGHPUT)
            rdma_buffer_size: RDMA buffer size in bytes (0 = auto)
            num_qp_per_rank: Number of RDMA queue pairs per rank (0 = auto)
            num_channels: Number of communication channels (0 = auto)
            use_pytorch_allocator: Use PyTorch's caching allocator for internal
                                   buffers (recommended for efficiency)
        """
        self.group = group
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.max_tokens_per_rank = max_tokens_per_rank
        self.algorithm = algorithm
        self.use_pytorch_allocator = use_pytorch_allocator

        # HT mode requires max_tokens_per_rank > 0 (no NCCL_EP_AUTO); see ep_test.cu and nccl_ep.cc
        if algorithm == ncclEpAlgorithm_t.NCCL_EP_ALGO_HIGH_THROUGHPUT and max_tokens_per_rank <= 0:
            raise ValueError(
                "NCCL EP HIGH_THROUGHPUT requires max_tokens_per_rank > 0 (per-rank send count). "
                "Dynamic max_tokens_per_rank (NCCL_EP_AUTO) is not supported. "
                "Set moe_nccl_ep_max_tokens_per_rank to the max tokens sent per rank (e.g. 512)."
            )

        # Validate algorithm
        if algorithm != ncclEpAlgorithm_t.NCCL_EP_ALGO_HIGH_THROUGHPUT:
            raise NotImplementedError(
                f"Only NCCL_EP_ALGO_HIGH_THROUGHPUT is currently supported. "
                f"Got algorithm={algorithm}. LOW_LATENCY mode is not yet implemented."
            )

        # Initialize NCCL library wrapper
        self.nccl = NCCLLibrary()

        if not self.nccl.ep_available:
            raise RuntimeError(
                "NCCL EP extensions not available. "
                "Please rebuild NCCL with MoE support."
            )

        # Get rank and device information
        self.device = torch.cuda.current_device()

        if group is None and 'OMPI_COMM_WORLD_RANK' in os.environ:
            # MPI-only mode without PyTorch distributed
            self.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            self.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        else:
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)

        self.num_local_experts = num_experts // self.world_size

        # Create NCCL communicator
        with torch.cuda.device(self.device):
            self.nccl_comm = get_nccl_comm_from_group(group, self.nccl)

            # Warmup: all-reduce to ensure communicator is initialized
            warmup_stream = torch.cuda.current_stream()
            warmup_data = torch.zeros(1, device=self.device)
            warmup_out = torch.empty_like(warmup_data)
            self.nccl.ncclAllReduce(
                warmup_data.data_ptr(),
                warmup_out.data_ptr(),
                warmup_data.numel(),
                ncclDataTypeEnum.from_torch(warmup_data.dtype),
                0,  # ncclSum
                self.nccl_comm,
                warmup_stream.cuda_stream
            )
            warmup_stream.synchronize()
            del warmup_data, warmup_out

            # Dedicated communication stream
            self._comm_stream = torch.cuda.Stream()

            # Store stream handle for MoE operations (will use comm stream)
            self.stream = self._comm_stream.cuda_stream

        # Create MoE group
        group_config = ncclEpGroupConfig_t()
        group_config.version = 1
        group_config.algorithm = algorithm
        group_config.num_experts = num_experts
        group_config.max_tokens_per_rank = max_tokens_per_rank
        group_config.token_size_bytes = hidden_dim * 2  # fp16/bf16
        group_config.rdma_buffer_size = rdma_buffer_size
        group_config.num_qp_per_rank = num_qp_per_rank
        group_config.num_channels = num_channels

        with torch.cuda.device(self.device):
            # Get PyTorch allocator callbacks if requested
            if use_pytorch_allocator:
                alloc_fn, free_fn = _get_pytorch_allocator_callbacks()
            else:
                alloc_fn, free_fn = None, None

            # Create MoE group (allocates workspace, NVLink, and RDMA buffers)
            self.ep_group = self.nccl.ncclEpCreateGroup(
                self.nccl_comm, group_config, self.stream,
                alloc_fn=alloc_fn, free_fn=free_fn
            )

        self.current_handle = None

        # Log buffer initialization
        logger.info(
            f"{_nccl_log_prefix()} NCCL EP INFO NCCLEPBuffer rank {self.rank} nRanks {self.world_size} "
            f"cudaDev {self.device} numExperts {num_experts} hiddenDim {hidden_dim} "
            f"algo {'HT' if algorithm == ncclEpAlgorithm_t.NCCL_EP_ALGO_HIGH_THROUGHPUT else 'LL'} "
            f"pytorchAlloc={use_pytorch_allocator} - Init COMPLETE"
        )

        # Store handle metadata: {handle_id: {'num_tokens': int, 'top_k': int}}
        self._handle_metadata = {}

        # Deferred destruction queue: [(handle, event)]
        # Handles are destroyed only after their associated events complete
        self._deferred_destruction_queue = []

        # Pre-allocate wrappers for dispatch (max 4 inputs, 4 outputs)
        self._dispatch_in_wrappers = [NDTensorWrapper(max_ndim=2) for _ in range(4)]
        self._dispatch_out_wrappers = [NDTensorWrapper(max_ndim=2) for _ in range(4)]

        # Pre-allocate wrappers for combine (max 2 inputs, 2 outputs)
        self._combine_in_wrappers = [NDTensorWrapper(max_ndim=2) for _ in range(2)]
        self._combine_out_wrappers = [NDTensorWrapper(max_ndim=2) for _ in range(2)]

        # Pre-allocate wrapper for handle creation
        self._handle_topk_wrapper = NDTensorWrapper(max_ndim=2)

        # Pre-allocate pointer arrays
        self._dispatch_in_ptrs = (ctypes.POINTER(ncclNDTensor_t) * 4)()
        self._dispatch_out_ptrs = (ctypes.POINTER(ncclNDTensor_t) * 4)()
        self._combine_in_ptrs = (ctypes.POINTER(ncclNDTensor_t) * 2)()
        self._combine_out_ptrs = (ctypes.POINTER(ncclNDTensor_t) * 2)()

        # Pre-allocate config structs
        self._dispatch_config = ncclEpDispatchConfig_t()
        self._dispatch_config.round_scales = 0

        # Pinned+mapped host memory for recv_expert_counter (expert counts written directly during handle creation)
        self._recv_expert_counter_size = self.num_local_experts
        self._recv_expert_counter_ptr = self._cuda_host_alloc_mapped(self._recv_expert_counter_size * 4)
        self._recv_expert_counter_array = (ctypes.c_int * self._recv_expert_counter_size).from_address(
            self._recv_expert_counter_ptr
        )
        self._recv_expert_counter_wrapper = NDTensorWrapper(max_ndim=1)

    def _allocate_recv_buffer(
        self, num_tokens: int, hidden_dim: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Allocate output buffer for dispatch (fresh tensor)."""
        return torch.empty((num_tokens, hidden_dim), dtype=dtype, device=device)

    def _allocate_combined_buffer(
        self, num_tokens: int, hidden_dim: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Allocate output buffer for combine (fresh tensor)."""
        return torch.empty((num_tokens, hidden_dim), dtype=dtype, device=device)

    def get_comm_stream(self) -> torch.cuda.Stream:
        """Get the communication stream.

        Returns:
            The communication stream.
        """
        return self._comm_stream

    @staticmethod
    def capture() -> "NCCLEPEventOverlap":
        """Capture a CUDA event on the current stream.

        Returns:
            The captured event.
        """
        return NCCLEPEventOverlap(torch.cuda.current_stream())

    @staticmethod
    def _stream_wait_event(stream: torch.cuda.Stream, event: "NCCLEPEventOverlap"):
        """Make stream wait for an event."""
        stream.wait_event(event.event)

    @staticmethod
    def _stream_wait_stream(waiting_stream: torch.cuda.Stream, other_stream: torch.cuda.Stream):
        """Make one stream wait for another."""
        waiting_stream.wait_stream(other_stream)

    @staticmethod
    def _cuda_host_alloc_mapped(size_bytes: int) -> int:
        """Allocate pinned host memory with cudaHostAllocMapped (required for cudaHostGetDevicePointer)."""
        cudart = _get_cudart()
        ptr = ctypes.c_void_p()
        result = cudart.cudaHostAlloc(ctypes.byref(ptr), ctypes.c_size_t(size_bytes), ctypes.c_uint(2))
        if result != 0:
            raise RuntimeError(f"cudaHostAlloc failed with error {result}")
        return ptr.value

    @staticmethod
    def _cuda_free_host(ptr: int):
        """Free pinned host memory allocated with cudaHostAlloc."""
        cudart = _get_cudart()
        result = cudart.cudaFreeHost(ctypes.c_void_p(ptr))
        if result != 0:
            raise RuntimeError(f"cudaFreeHost failed with error {result}")

    def __del__(self):
        """Cleanup NCCL EP resources."""
        try:
            if hasattr(self, 'current_handle') and self.current_handle is not None:
                self.nccl.ncclEpHandleDestroy(self.current_handle)
            if hasattr(self, 'ep_group') and self.ep_group:
                self.nccl.ncclEpGroupDestroy(self.ep_group, self.stream)
            if hasattr(self, '_recv_expert_counter_ptr') and self._recv_expert_counter_ptr:
                self._cuda_free_host(self._recv_expert_counter_ptr)
        except Exception:
            pass  # Best effort cleanup

    def dispatch(
        self,
        x: torch.Tensor,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        use_fp8: bool = False,
        handle: Optional[object] = None,
        previous_event: Optional["NCCLEPEventOverlap"] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], object, "NCCLEPEventOverlap"]:
        """Dispatch tokens to experts using NCCL EP.

        Args:
            x: Input tensor [num_tokens, hidden_dim]
            topk_idx: Token routing indices [num_tokens, topk]. Required when handle=None.
                     When handle is provided, routing info is retrieved from the handle.
            topk_weights: Token routing weights [num_tokens, topk].
                         Required when creating a new handle (handle=None).
                         Not needed for cached dispatch (when handle is provided).
            use_fp8: Whether to use FP8 precision. Only used when handle=None.
                    When handle is provided, use_fp8 is retrieved from the handle's config.
            handle: Optional pre-existing handle for cached dispatch mode (e.g., backward pass).
                   If None, creates a new handle (requires topk_idx and topk_weights).
                   When handle is provided, routing info is reused from the handle.
            previous_event: Event to wait for before executing.
                           Used for chaining async operations.
            async_finish: If True, compute stream won't wait for comm to finish
                         (ignored; NCCL EP runs in sync-only mode).
            allocate_on_comm_stream: If True, allocate tensors on comm stream
                         (ignored; NCCL EP runs in sync-only mode).

        Returns:
            Tuple of:
                - recv_x: Received tokens after dispatch
                - recv_token_indices: Received token indices (None in cached mode)
                - recv_token_probs: Received token probabilities (None in cached mode)
                - handle: NCCL EP handle (either the provided one or newly created)
                - event: NCCLEPEventOverlap for async completion tracking
        """
        # NCCL EP: sync-only; async params kept for API compatibility.
        async_finish = False
        allocate_on_comm_stream = False

        # Validate parameters based on whether we're creating a new handle
        if handle is None:
            # Creating new handle: require both topk_idx and topk_weights
            if topk_idx is None:
                raise ValueError(
                    "topk_idx is required when creating a new handle (handle=None)."
                )
            if topk_weights is None:
                raise ValueError("topk_weights is required when creating a new handle.")
        else:
            # Reusing handle (cached mode):
            # - topk_idx: Retrieved from handle->topk_idx (C++ API), no need to pass
            # - use_fp8: Retrieved from handle->config.use_fp8 (parameter ignored)
            # - topk_weights: Optional in cached mode (C++ test shows 1 input only)
            #   Following the C++ test pattern from multiproc_singlegpu_moe.cu:
            #   Second dispatch uses only 1 input (tokens) and 1 output (recv_tokens)
            handle_id = handle.value if isinstance(handle, ctypes.c_void_p) else id(handle)
            is_cached_dispatch = (handle_id in self._handle_metadata and
                                  self._handle_metadata[handle_id].get('dispatch_done', False))
            if topk_weights is None and not is_cached_dispatch:
                raise ValueError(
                    "topk_weights is required for the first dispatch with a reused handle. "
                    "Subsequent dispatches (cached mode) can omit topk_weights."
                )

        # allocate_on_comm_stream requires previous_event and async_finish
        if allocate_on_comm_stream:
            assert previous_event is not None and async_finish, \
                "allocate_on_comm_stream requires previous_event and async_finish=True"

        with torch.cuda.device(self.device):
            # Clean up any handles from previous iterations that have completed (non-blocking check)
            self._process_deferred_destructions()

            compute_stream = torch.cuda.current_stream()

            if allocate_on_comm_stream:
                torch.cuda.set_stream(self._comm_stream)

            # Comm stream waits for previous_event or compute_stream
            if previous_event is not None:
                self._stream_wait_event(self._comm_stream, previous_event)
            else:
                self._stream_wait_stream(self._comm_stream, compute_stream)

            self.stream = self._comm_stream.cuda_stream

            # Create or reuse handle
            if handle is None:
                # Configure pre-allocated wrapper for topk_idx
                self._handle_topk_wrapper.configure(
                    topk_idx,
                    tag=ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_IDX
                )

                # Configure recv_expert_counter wrapper (mapped host buffer for direct expert count writes)
                self._recv_expert_counter_wrapper.nd_tensor.ndim = 1
                self._recv_expert_counter_wrapper._sizes[0] = self._recv_expert_counter_size
                self._recv_expert_counter_wrapper._strides[0] = 1
                self._recv_expert_counter_wrapper.nd_tensor.datatype = ncclDataTypeEnum.ncclInt32
                self._recv_expert_counter_wrapper.nd_tensor.data = self._recv_expert_counter_ptr
                self._recv_expert_counter_wrapper.nd_tensor.tag = ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST
                self._recv_expert_counter_wrapper.nd_tensor.flags = 0

                # Create handle for this dispatch (will be reused for combine and backward pass)
                # This also populates recv_expert_counter via metadata exchange

                # Prepare local_tensors array with recv_expert_counter
                local_tensors = [self._recv_expert_counter_wrapper.nd_tensor]

                self.current_handle = self.nccl.ncclEpCreateHandle(
                    self.ep_group,
                    self._handle_topk_wrapper.nd_tensor,
                    None,  # config must be None/NULL (reserved for future use)
                    self.stream,
                    local_tensors=local_tensors,
                    use_fp8=use_fp8
                )

                # WORKAROUND: Synchronize stream after handle creation to ensure GPU kernel
                # has written expert counts to mapped memory before CPU reads them.
                # The C++ code polls mapped memory but doesn't sync the stream first,
                # which can cause stale reads in busy GPU environments.
                if not async_finish:
                    # Sync mode: synchronize stream to ensure GPU writes are CPU-visible
                    # Expert counts are written by GPU regardless of max_tokens_per_rank value
                    self._comm_stream.synchronize()
                    # Read expert counts immediately since stream is now synced
                    expert_counts_snapshot = list(self._recv_expert_counter_array)
                else:
                    # Async mode: record an event for deferred expert count retrieval
                    self._handle_creation_event = torch.cuda.Event()
                    self._handle_creation_event.record(self._comm_stream)
                    expert_counts_snapshot = None  # Will be populated on-demand

                # Store metadata for this handle (for shape info when reusing)
                handle_id = self.current_handle.value if isinstance(self.current_handle, ctypes.c_void_p) else id(self.current_handle)
                # num_recv_tokens = nRanks * max_tokens_per_rank
                forward_num_recv_tokens = self.nccl.ncclEpHandleGetNumRecvTokens(self.current_handle)
                self._handle_metadata[handle_id] = {
                    'num_tokens': topk_idx.shape[0],
                    'top_k': topk_idx.shape[1],
                    'num_recv_tokens': forward_num_recv_tokens,
                    'expert_counts': expert_counts_snapshot,  # Snapshot taken after sync (if applicable)
                    'expert_counts_synced': expert_counts_snapshot is not None,  # True if we already read them
                }
            else:
                # Reuse existing handle (e.g., for backward pass)
                self.current_handle = handle
                handle_id = handle.value if isinstance(handle, ctypes.c_void_p) else id(handle)

            # Get handle_id for metadata lookup
            handle_id = self.current_handle.value if isinstance(self.current_handle, ctypes.c_void_p) else id(self.current_handle)

            # Determine number of tokens for buffer allocation
            if topk_idx is not None:
                num_tokens = topk_idx.shape[0]
                top_k = topk_idx.shape[1]
            elif handle is not None:
                # Get shape from handle's stored metadata (from handle->topk_idx)
                if handle_id in self._handle_metadata:
                    metadata = self._handle_metadata[handle_id]
                    num_tokens = metadata['num_tokens']
                    top_k = metadata['top_k']
                else:
                    raise ValueError(
                        "Cannot determine tensor shape from handle: handle metadata not found. "
                        "Please pass topk_weights or topk_idx to determine buffer allocation size."
                    )
            else:
                raise ValueError(
                    "Cannot determine tensor shape: topk_weights, topk_idx, and handle are all None. "
                    "At least one must be provided to determine buffer allocation size."
                )

            # Check if this is a cached dispatch (handle was provided AND dispatch was done before)
            is_cached_dispatch = (handle is not None and
                                  handle_id in self._handle_metadata and
                                  self._handle_metadata[handle_id].get('dispatch_done', False))

            if is_cached_dispatch:
                num_recv_tokens = self._handle_metadata[handle_id]['num_recv_tokens']
            else:
                num_recv_tokens = self.nccl.ncclEpHandleGetNumRecvTokens(self.current_handle)

            # Allocate output tensor (always fresh allocation)
            recv_x = self._allocate_recv_buffer(num_recv_tokens, x.shape[1], x.dtype, x.device)

            # Build input tensor list using pre-allocated wrappers
            num_inputs = 0

            # Input: x (always present)
            self._dispatch_in_wrappers[num_inputs].configure(
                x, tag=ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS
            )
            self._dispatch_in_ptrs[num_inputs] = self._dispatch_in_wrappers[num_inputs].get_pointer()
            num_inputs += 1

            # Build output tensor list using pre-allocated wrappers
            num_outputs = 0

            # Output: recv_x (always present)
            self._dispatch_out_wrappers[num_outputs].configure(
                recv_x, tag=ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS
            )
            self._dispatch_out_ptrs[num_outputs] = self._dispatch_out_wrappers[num_outputs].get_pointer()
            num_outputs += 1

            recv_token_indices = None
            recv_token_probs = None

            if not is_cached_dispatch:
                # First dispatch: also pass topk_weights and topk_idx as inputs
                self._dispatch_in_wrappers[num_inputs].configure(
                    topk_weights,
                    tag=ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS
                )
                self._dispatch_in_ptrs[num_inputs] = self._dispatch_in_wrappers[num_inputs].get_pointer()
                num_inputs += 1

                self._dispatch_in_wrappers[num_inputs].configure(
                    topk_idx,
                    tag=ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_IDX
                )
                self._dispatch_in_ptrs[num_inputs] = self._dispatch_in_wrappers[num_inputs].get_pointer()
                num_inputs += 1

                # First dispatch: also output recv_topk_weights and recv_topk_idx
                recv_token_probs = torch.empty(
                    (num_recv_tokens, top_k), dtype=topk_weights.dtype, device=topk_weights.device
                )
                self._dispatch_out_wrappers[num_outputs].configure(
                    recv_token_probs,
                    tag=ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS
                )
                self._dispatch_out_ptrs[num_outputs] = self._dispatch_out_wrappers[num_outputs].get_pointer()
                num_outputs += 1

                recv_token_indices = torch.empty(
                    (num_recv_tokens, top_k), dtype=topk_idx.dtype, device=topk_idx.device
                )
                self._dispatch_out_wrappers[num_outputs].configure(
                    recv_token_indices,
                    tag=ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_IDX
                )
                self._dispatch_out_ptrs[num_outputs] = self._dispatch_out_wrappers[num_outputs].get_pointer()
                num_outputs += 1

            # Execute dispatch (HT mode: no local tensors)
            self.nccl.ncclEpDispatch(
                self.current_handle,
                self._dispatch_in_ptrs, num_inputs,
                self._dispatch_out_ptrs, num_outputs,
                None, 0,  # No local tensors in HT mode
                0,  # send_only = False
                self._dispatch_config,
                self.stream
            )

            # Mark dispatch done for cached dispatch detection
            if handle_id in self._handle_metadata:
                self._handle_metadata[handle_id]['dispatch_done'] = True

            # Stream synchronization
            if async_finish:
                event = NCCLEPEventOverlap(self._comm_stream)
                # Record tensors on streams to prevent early deallocation
                for t in [x, recv_x]:
                    if t is not None:
                        t.record_stream(self._comm_stream)
                        if allocate_on_comm_stream:
                            t.record_stream(compute_stream)

                for t in [topk_idx, topk_weights, recv_token_indices, recv_token_probs]:
                    if t is not None:
                        t.record_stream(self._comm_stream)
                        if allocate_on_comm_stream:
                            t.record_stream(compute_stream)
            else:
                # Sync mode: compute waits for comm
                self._stream_wait_stream(compute_stream, self._comm_stream)
                event = NCCLEPEventOverlap(compute_stream)

            self._expert_counts_available = not is_cached_dispatch

            if allocate_on_comm_stream:
                torch.cuda.set_stream(compute_stream)

        # Log dispatch operation
        logger.debug(
            f"{_nccl_log_prefix()} NCCL EP INFO ncclEpDispatch "
            f"inputTokens {x.shape[0]} outputTokens {recv_x.shape[0]} hiddenDim {x.shape[1]} "
            f"handle 0x{handle_id:x} cached={is_cached_dispatch}"
        )

        return (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            self.current_handle,
            event
        )

    def get_tokens_per_expert_list(self) -> list:
        """Get the number of tokens per expert as a Python list.

        When recv_expert_counter is passed to ncclEpCreateHandle, the library fills it during
        metadata preprocessing with per-expert token counts for this rank. This buffer always
        passes that tensor at handle creation, so after the first dispatch (non-cached) a valid
        list of length num_local_experts is available.
        Counts returned are for the current handle (self.current_handle, i.e. the handle from
        the most recent dispatch()). Call after dispatch (and optional sync) so the stream has
        completed and counts are readable.

        Returns:
            List of token counts per expert, or empty list if counts are not available
            (e.g., cached dispatch mode with reused handle).
        """
        if not getattr(self, '_expert_counts_available', False):
            return []

        if self.current_handle is not None:
            handle_id = self.current_handle.value if isinstance(self.current_handle, ctypes.c_void_p) else id(self.current_handle)

            if handle_id in self._handle_metadata:
                metadata = self._handle_metadata[handle_id]

                # Lazy reading of expert counts
                if not metadata.get('expert_counts_synced', False):
                    # In async mode, synchronize the event before reading
                    if hasattr(self, '_handle_creation_event'):
                        self._handle_creation_event.synchronize()
                        delattr(self, '_handle_creation_event')
                    # Now safe to read expert counts from mapped host memory
                    expert_counts_snapshot = list(self._recv_expert_counter_array)
                    metadata['expert_counts'] = expert_counts_snapshot
                    metadata['expert_counts_synced'] = True

                if 'expert_counts' in metadata and metadata['expert_counts'] is not None:
                    return metadata['expert_counts']

        # Fallback to shared buffer (shouldn't happen in normal operation)
        # In async mode, we need to sync the event first
        if hasattr(self, '_handle_creation_event'):
            self._handle_creation_event.synchronize()
            delattr(self, '_handle_creation_event')
        return list(self._recv_expert_counter_array)

    def combine(
        self,
        x: torch.Tensor,
        handle: object,
        topk_weights: Optional[torch.Tensor] = None,
        previous_event: Optional["NCCLEPEventOverlap"] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], "NCCLEPEventOverlap"]:
        """Combine expert outputs using NCCL EP.

        Args:
            x: Expert output tensor [num_recv_tokens, hidden]
            handle: NCCL EP handle from dispatch
            topk_weights: Optional routing weights [num_recv_tokens, top_k].
                         Not needed in forward pass.
                         Used in backward pass (dispatch.backward) for gradient computation.
                         When provided, combined_topk_weights will also be returned.
            previous_event: Event to wait for before executing.
                           Used for chaining async operations.
            async_finish: If True, compute stream won't wait for comm to finish
                         (ignored; NCCL EP runs in sync-only mode).
            allocate_on_comm_stream: If True, allocate tensors on comm stream
                         (ignored; NCCL EP runs in sync-only mode).

        Returns:
            Tuple of:
                - combined_x: Combined output tensor [num_tokens, hidden]
                - combined_topk_weights: Combined weights [num_tokens, top_k] or None
                - event: NCCLEPEventOverlap for async completion tracking
        """
        # NCCL EP: sync-only; async params kept for API compatibility.
        async_finish = False
        allocate_on_comm_stream = False

        # allocate_on_comm_stream requires previous_event and async_finish
        if allocate_on_comm_stream:
            assert previous_event is not None and async_finish, \
                "allocate_on_comm_stream requires previous_event and async_finish=True"

        with torch.cuda.device(self.device):
            compute_stream = torch.cuda.current_stream()

            if allocate_on_comm_stream:
                torch.cuda.set_stream(self._comm_stream)

            # Comm stream waits for previous_event or compute_stream
            if previous_event is not None:
                self._stream_wait_event(self._comm_stream, previous_event)
            else:
                self._stream_wait_stream(self._comm_stream, compute_stream)

            self.stream = self._comm_stream.cuda_stream

            handle_id = handle.value if isinstance(handle, ctypes.c_void_p) else id(handle)

            combined_topk_weights = None

            # Get shape from handle metadata
            if handle_id not in self._handle_metadata:
                raise ValueError(
                    f"Handle metadata not found for handle_id {handle_id}. "
                    "Cannot determine combine output shape."
                )
            original_num_tokens = self._handle_metadata[handle_id]['num_tokens']
            top_k = self._handle_metadata[handle_id]['top_k']
            hidden_dim = x.shape[1]

            # Allocate output tensor
            combined_x = self._allocate_combined_buffer(original_num_tokens, hidden_dim, x.dtype, x.device)

            # Build input tensor list
            num_inputs = 0
            self._combine_in_wrappers[num_inputs].configure(
                x, tag=ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS
            )
            self._combine_in_ptrs[num_inputs] = self._combine_in_wrappers[num_inputs].get_pointer()
            num_inputs += 1

            # Build output tensor list
            num_outputs = 0
            self._combine_out_wrappers[num_outputs].configure(
                combined_x, tag=ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS
            )
            self._combine_out_ptrs[num_outputs] = self._combine_out_wrappers[num_outputs].get_pointer()
            num_outputs += 1

            if topk_weights is not None:
                # C++ HT combine expects topk_weights shape [num_recv_tokens, top_k] (one row per combine input x)
                if topk_weights.shape[0] != x.shape[0] or topk_weights.shape[1] != top_k:
                    raise ValueError(
                        f"topk_weights must have shape [num_recv_tokens, top_k] = [{x.shape[0]}, {top_k}], "
                        f"got {tuple(topk_weights.shape)}. HT combine expects one row per combine input token."
                    )
                # Input weights tensor
                self._combine_in_wrappers[num_inputs].configure(
                    topk_weights,
                    tag=ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS
                )
                self._combine_in_ptrs[num_inputs] = self._combine_in_wrappers[num_inputs].get_pointer()
                num_inputs += 1

                # Output combined weights tensor - shape [original_num_tokens, top_k]
                combined_topk_weights = torch.empty(
                    (original_num_tokens, top_k), dtype=topk_weights.dtype, device=topk_weights.device
                )
                self._combine_out_wrappers[num_outputs].configure(
                    combined_topk_weights,
                    tag=ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS
                )
                self._combine_out_ptrs[num_outputs] = self._combine_out_wrappers[num_outputs].get_pointer()
                num_outputs += 1

            # Execute combine (no local_tensors needed - weights are now input/output)
            self.nccl.ncclEpCombine(
                handle,
                self._combine_in_ptrs, num_inputs,
                self._combine_out_ptrs, num_outputs,
                None, 0,  # No local tensors
                0, None, self.stream
            )

            # Stream synchronization
            if async_finish:
                event = NCCLEPEventOverlap(self._comm_stream)
                # Record tensors on streams to prevent early deallocation
                for t in [x, combined_x]:
                    if t is not None:
                        t.record_stream(self._comm_stream)
                        if allocate_on_comm_stream:
                            t.record_stream(compute_stream)
                for t in [topk_weights, combined_topk_weights]:
                    if t is not None:
                        t.record_stream(self._comm_stream)
                        if allocate_on_comm_stream:
                            t.record_stream(compute_stream)
            else:
                # Sync mode: compute waits for comm
                self._stream_wait_stream(compute_stream, self._comm_stream)
                event = NCCLEPEventOverlap(compute_stream)

            if allocate_on_comm_stream:
                torch.cuda.set_stream(compute_stream)

        # Log combine operation
        logger.debug(
            f"{_nccl_log_prefix()} NCCL EP INFO ncclEpCombine "
            f"inputTokens {x.shape[0]} outputTokens {combined_x.shape[0]} hiddenDim {x.shape[1]} "
            f"handle 0x{handle_id:x}"
        )

        return combined_x, combined_topk_weights, event

    def _process_deferred_destructions(self):
        """Process deferred handle destructions for completed operations."""
        remaining = []
        for handle, event in self._deferred_destruction_queue:
            if event.query():  # Event completed, safe to destroy
                handle_id = handle.value if isinstance(handle, ctypes.c_void_p) else id(handle)
                self._handle_metadata.pop(handle_id, None)
                self.nccl.ncclEpHandleDestroy(handle)
                if self.current_handle == handle:
                    self.current_handle = None
            else:
                remaining.append((handle, event))
        self._deferred_destruction_queue = remaining

    def destroy_handle(self, handle: object):
        """Destroy a handle and free associated cached structures.

        This uses deferred destruction: the handle is queued for destruction and will be
        destroyed asynchronously once all GPU operations complete. This prevents blocking
        the training pipeline while ensuring safe destruction.

        Args:
            handle: The NCCL EP handle to destroy
        """
        if handle is not None:
            # Record an event on comm stream to track when operations complete
            with torch.cuda.device(self.device):
                event = NCCLEPEventOverlap(self._comm_stream)

            # Add to deferred destruction queue
            self._deferred_destruction_queue.append((handle, event))

            # Process any completed destructions (non-blocking check)
            self._process_deferred_destructions()
