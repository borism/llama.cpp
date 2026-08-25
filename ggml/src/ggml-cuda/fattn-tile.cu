#include "common.cuh"
#include "fattn-tile.cuh"

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    // Disabled on this board: this CUDA 10 / C++14 build cannot use
    // if-constexpr, and fattn-tile.cuh's per-(DKQ,DV,ncols1,ncols2) config
    // validity checks rely on it throughout -- confirmed broken even for
    // ordinary symmetric head dims, not just DeepSeek-MLA's asymmetric ones.
    // ggml_cuda_flash_attn_ext_supported() always returns false, so llama.cpp
    // uses its unfused (explicit softmax) attention path instead; this should
    // never actually be reached.
    GGML_ABORT("flash attention (tile kernel) is disabled in this CUDA 10 build");
}
