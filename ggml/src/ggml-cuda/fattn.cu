#include "fattn.cuh"

// Flash attention is disabled in this CUDA 10 / C++14 build (Jetson Nano, Maxwell CC 5.3):
// the fattn-{tile,mma-f16,vec} kernel dispatch relies on if-constexpr throughout and can't be
// ported to C++14 without risking silently wrong attention output. Reporting the op as
// unsupported makes llama.cpp fall back to its unfused attention path. fattn-tile.cu and the
// fattn template instances are excluded from the build (ggml-cuda/CMakeLists.txt), so the
// fattn-*.cuh headers stay pristine upstream and never get compiled.
//
// NOTE for RPC: ggml-rpc's client reports every op as supported and the server has no CPU
// fallback, so a head using this device over RPC must run with flash attention off (-fa off).

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("flash attention is disabled in this CUDA 10 build (use -fa off)");
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    return ggml_nbytes(dst);
}
