#include "allreduce.cuh"

// Stubbed for single-GPU builds (Jetson Nano / Maxwell, CUDA 10 patched build):
// the original multi-GPU tensor-parallel AllReduce implementation uses
// nv_bfloat16 conversion paths incompatible with this CUDA 10 shim, and is
// unreachable here anyway since this board only has one GPU. See
// allreduce.cu.orig for the real implementation.

struct ggml_cuda_ar_pipeline {};

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int *, size_t) {
    return nullptr;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline *) {}

bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}
