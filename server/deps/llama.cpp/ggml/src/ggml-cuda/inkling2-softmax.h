#pragma once

// Online-softmax update of the Inkling 2 global split-KV attention, shared by
// the kernel and the GPU-free unit test.
//
// With m_new = max(m_prev, score) exactly one of the two exponentials
//   exp(m_prev - m_new) and exp(score - m_new)
// has argument 0 and is therefore exactly 1.0f. A single exp of the non-zero
// argument, exp(-|score - m_prev|), therefore yields both factors and keeps
// alpha and p bit-identical to the two-exp reference while halving the expf
// calls. Measured in lucebox/benchmarks/inkling2-q8-single-exp.md (b50200a):
// -21.05% per layer at 65.514 keys, -17.39% on the global attention per step,
// same FP64 max error and identical tokens.
//
// Kept free of CUDA/HIP declarations: callers pass the exponential as a
// functor (expf on device, a counting shim in the test), so the host test can
// pin both the arithmetic and the number of exp calls.

#include <cmath>

// The header is also compiled by the plain host unit-test TU, which has no HIP
// headers; there the helper needs no host/device attribute.
#if defined(__HIPCC__) || defined(__CUDACC__)
#define INKLING2_SOFTMAX_HD __host__ __device__
#else
#define INKLING2_SOFTMAX_HD
#endif

namespace inkling2_softmax {

template <class ExpFn>
INKLING2_SOFTMAX_HD inline void online_softmax_pair(float m_prev, float score, ExpFn exp_fn, float & alpha, float & p) {
    // score > m_prev is the "raised" branch: p is exp(0) == 1 and alpha is the
    // non-zero one. exp(-inf) == 0 covers the first key, where m_prev == -inf
    // and the reference used an explicit isinf(m_prev) ? 0.0f guard; the guard
    // is kept so a +inf m_prev stays identical as well.
    const bool  raised = score > m_prev;
    const float e      = exp_fn(-std::fabs(score - m_prev));
    alpha = raised ? e : (std::isinf(m_prev) ? 0.0f : 1.0f);
    p     = raised ? 1.0f : e;
}

} // namespace inkling2_softmax

#undef INKLING2_SOFTMAX_HD
