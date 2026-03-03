# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""NCCL EP backend for high-throughput MoE token dispatching.

Pure Python (ctypes) implementation; no compilation needed. Works with LD_PRELOAD
setup. Only HIGH_THROUGHPUT algorithm is supported.
"""

try:
    from nccl_ep.nccl_wrapper import (
        NCCLLibrary,
        ncclEpAlgorithm_t,
    )
    from .buffer import NCCLEPBuffer

    HAVE_NCCL_EP = True
    NCCL_EP_ALGO_LOW_LATENCY = ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY  # Not yet supported
    NCCL_EP_ALGO_HIGH_THROUGHPUT = ncclEpAlgorithm_t.NCCL_EP_ALGO_HIGH_THROUGHPUT

except ImportError as e:
    HAVE_NCCL_EP = False
    _import_error = str(e)
    NCCLEPBuffer = None
    NCCL_EP_ALGO_LOW_LATENCY = 0
    NCCL_EP_ALGO_HIGH_THROUGHPUT = 1

__all__ = [
    'HAVE_NCCL_EP',
    'NCCLEPBuffer',
    'NCCL_EP_ALGO_LOW_LATENCY',
    'NCCL_EP_ALGO_HIGH_THROUGHPUT',
]

if not HAVE_NCCL_EP:
    import warnings
    warnings.warn(
        f"NCCL EP backend is not available. Error: {_import_error}\n"
        "Make sure:\n"
        "  1. NCCL_HOME points to your NCCL EP build.\n"
        "  2. LD_PRELOAD is set to force PyTorch to use custom NCCL.\n"
        "  3. NCCL library has MoE extensions.\n"
        "Use deepep or hybridep backend instead, or standard alltoall dispatcher."
    )
