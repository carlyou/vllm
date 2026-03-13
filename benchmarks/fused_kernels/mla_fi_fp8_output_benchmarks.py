# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark FlashInfer MLA prefill with/without native FP8 output.

Compares:
  1. bf16 output + separate FP8 quant kernel  (baseline)
  2. Native FP8 output via o_scale            (fused)

Usage:
    python benchmarks/fused_kernels/mla_fi_fp8_output_benchmarks.py
"""

import torch
import torch.utils.benchmark as TBenchmark

from vllm.platforms import current_platform

# DeepSeek-V3 MLA dimensions
NUM_QO_HEADS = 128
NUM_KV_HEADS = 1
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
HEAD_DIM_QK = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
HEAD_DIM_VO = 128

FP8_DTYPE = current_platform.fp8_dtype()
BF16 = torch.bfloat16
DEVICE = torch.device("cuda:0")


def make_inputs(num_tokens: int):
    """Create random q, k, v tensors for MLA prefill."""
    q = torch.randn(num_tokens, NUM_QO_HEADS, HEAD_DIM_QK, dtype=BF16, device=DEVICE)
    k = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM_QK, dtype=BF16, device=DEVICE)
    v = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM_VO, dtype=BF16, device=DEVICE)
    return q, k, v


def run_bf16_then_quant(wrapper, q, k, v, output_scale: float):
    """Baseline: bf16 output from FlashInfer, then separate FP8 quant."""
    out_bf16 = wrapper.run(q=q, k=k, v=v, return_lse=False)
    # Simulate the separate quant kernel
    out_fp8 = (out_bf16 * output_scale).to(FP8_DTYPE)
    return out_fp8


def run_native_fp8(wrapper, q, k, v, output_scale: float):
    """Fused: FlashInfer native FP8 output via o_scale."""
    out = torch.empty(
        q.shape[0], q.shape[1], v.shape[2], device=DEVICE, dtype=FP8_DTYPE
    )
    result = wrapper.run(
        q=q, k=k, v=v, return_lse=False, out=out, o_scale=1.0 / output_scale
    )
    return result


def benchmark_one(num_tokens: int, num_seqs: int):
    """Benchmark a single configuration."""
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

    q, k, v = make_inputs(num_tokens)
    output_scale = 0.01  # typical FP8 scale

    # Build indptr for equal-length sequences
    seq_len = num_tokens // num_seqs
    qo_indptr = torch.arange(
        0, num_tokens + 1, seq_len, dtype=torch.int32, device=DEVICE
    )
    kv_indptr = qo_indptr.clone()

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    # --- Plan with bf16 output (CUTLASS backend for apples-to-apples) ---
    wrapper_bf16 = BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="cutlass"
    )
    wrapper_bf16.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM_QK,
        head_dim_vo=HEAD_DIM_VO,
        causal=True,
        sm_scale=1.0 / (HEAD_DIM_QK**0.5),
        q_data_type=BF16,
        o_data_type=BF16,
    )

    # --- Plan with FP8 output (CUTLASS backend required for FP8) ---
    wrapper_fp8 = BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="cutlass"
    )
    wrapper_fp8.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM_QK,
        head_dim_vo=HEAD_DIM_VO,
        causal=True,
        sm_scale=1.0 / (HEAD_DIM_QK**0.5),
        q_data_type=BF16,
        o_data_type=FP8_DTYPE,
    )

    label = f"tokens={num_tokens}, seqs={num_seqs}"

    # Warmup
    for _ in range(5):
        run_bf16_then_quant(wrapper_bf16, q, k, v, output_scale)
        run_native_fp8(wrapper_fp8, q, k, v, output_scale)

    t_baseline = TBenchmark.Timer(
        stmt="run_bf16_then_quant(wrapper, q, k, v, output_scale)",
        globals={
            "run_bf16_then_quant": run_bf16_then_quant,
            "wrapper": wrapper_bf16,
            "q": q,
            "k": k,
            "v": v,
            "output_scale": output_scale,
        },
        label=label,
        sub_label="bf16 + quant",
    ).blocked_autorange(min_run_time=2.0)

    t_fused = TBenchmark.Timer(
        stmt="run_native_fp8(wrapper, q, k, v, output_scale)",
        globals={
            "run_native_fp8": run_native_fp8,
            "wrapper": wrapper_fp8,
            "q": q,
            "k": k,
            "v": v,
            "output_scale": output_scale,
        },
        label=label,
        sub_label="native fp8",
    ).blocked_autorange(min_run_time=2.0)

    return t_baseline, t_fused


def main():
    torch.manual_seed(42)

    configs = [
        # (num_tokens, num_seqs)
        (128, 1),
        (256, 1),
        (512, 1),
        (1024, 1),
        (2048, 1),
        (4096, 1),
        (256, 4),
        (1024, 4),
        (4096, 4),
    ]

    results = []
    for num_tokens, num_seqs in configs:
        print(f"Benchmarking tokens={num_tokens}, seqs={num_seqs}...")
        t_baseline, t_fused = benchmark_one(num_tokens, num_seqs)
        results.extend([t_baseline, t_fused])

    compare = TBenchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    main()
