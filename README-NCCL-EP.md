# Megatron-LM with NCCL EP

NCCL EP integration for Megatron-LM: Flex token-dispatcher backend for MoE using the NCCL EP API for token dispatch and combine.

---

## Release notes

**Version:** Based on Megatron-LM / Megatron Core 0.12.0, with modifications to support the **NCCL-EP API** for MoE token dispatching.

### Supported today

- **Status:** Beta.
- **Scope:** MoE token dispatch/combine (intranode and internode) via the NCCL EP API.
- **Integration:** Flex token dispatcher, backend `ncclep`: `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend ncclep`.
- **Algorithm:** NCCL EP HIGH_THROUGHPUT only.
- **Precision:** BF16.

### Limitations and known issues

- **Sync mode only.** Synchronous execution only. Async dispatch/combine and overlap are disabled; `async_finish` and `allocate_on_comm_stream` are ignored (API compatibility only).
- **NCCL EP:** For further limitations and supported platforms, see the NCCL EP repository documentation.

---

## How to run

**Prerequisites:** NCCL build with MoE extensions (e.g. NCCL 2.29.x). Set the following so the process and PyTorch use this build:

```bash
export NCCL_HOME=/path/to/nccl/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$NCCL_HOME/lib/libnccl.so${LD_PRELOAD:+:$LD_PRELOAD}
```

**Minimal training args:**

```bash
--moe-token-dispatcher-type flex \
--moe-flex-dispatcher-backend ncclep \
--moe-nccl-ep-max-tokens-per-rank <max_tokens_per_rank>
```

Set `max_tokens_per_rank` ≥ max tokens per rank in the EP group (e.g. `(seq_length / (tp * cp)) * micro_batch_size`). Optional: `--moe-nccl-ep-num-channels`, `--moe-nccl-ep-num-qp-per-rank` (see `transformer_config.py`).
