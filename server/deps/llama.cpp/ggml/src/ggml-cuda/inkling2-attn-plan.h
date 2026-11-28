#pragma once

// Host-side launch plan for the global-layer (non-SWA) split-KV attention of
// the Inkling 2 verify pass: GGML_INKLING2_ATTN_REL with swa_window == 0 and
// T <= 16 (wider widths take the prefill/WMMA paths).
//
// Kept free of CUDA/HIP declarations so the host unit tests can pin the split
// count and the partial-buffer footprint without a GPU.

#include <cstddef>

namespace inkling2_attn_plan {

// Floats per (split, token, head) partial: 128 output values + max + sum.
constexpr std::size_t kPartialStride = 130;

// KV splits for the global attention of the verify pass.
//
// 16 splits for the whole narrow verify bucket (T <= 8: the DSpark block at
// T == 8 plus the short tails the fixtures exercise), 4 for wider decode.
// Evidence: lucebox/benchmarks/inkling2-q8-splits16.md (eb28ccd) measures 16
// splits at T == 8 (split+reduce -11% isolated, -4.4/-4.8% in-process, FP64
// error halved, identical tokens); ...-e2e.md (8c52c64) measures +2.02%/+1.80%
// end-to-end decode with token parity; isolation 7d6fddd identifies the kernel.
// The measurement also covered the width-6 tail, which stayed token-identical
// and was 3.5% faster on its global attention, so the whole T <= 8 bucket is
// adopted rather than special-casing T == 8.
inline int global_split_count(int n_tokens) {
    return n_tokens <= 2 ? 16 : (n_tokens <= 8 ? 16 : 4);
}

// Elements of the float partial buffer. This must be derived from the same
// split count the kernel is launched with: a fixed 8 here would turn a 16-split
// launch into a silent out-of-bounds write.
inline std::size_t partial_buffer_elems(int n_tokens, int n_head_q, int n_splits) {
    return (std::size_t) n_splits * (std::size_t) n_tokens * (std::size_t) n_head_q * kPartialStride;
}

} // namespace inkling2_attn_plan
