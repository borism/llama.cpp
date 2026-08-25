#pragma once
#include "common.cuh"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template<typename T>
using to_t_cuda_t = void (*)(const void * x, T * y, int64_t k, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;
typedef to_t_cuda_t<nv_bfloat16> to_bf16_cuda_t;

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type);

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

// TODO more general support for non-contiguous inputs

template<typename T>
using to_t_nc_cuda_t = void (*)(const void * x, T * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

typedef to_t_nc_cuda_t<float> to_fp32_nc_cuda_t;
typedef to_t_nc_cuda_t<half> to_fp16_nc_cuda_t;
typedef to_t_nc_cuda_t<nv_bfloat16> to_bf16_nc_cuda_t;

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type);
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type);
to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type);

// Rewritten from the upstream if-constexpr chain as mutually exclusive
// SFINAE overloads: this CUDA 10 / C++14 build has no if-constexpr, and a
// plain runtime if would require every branch (including ones assuming
// unrelated concrete types, e.g. float2 members on a scalar float) to
// type-check for every instantiation. Overloads sidestep that: an
// unselected overload's body is never instantiated at all.
template<typename dst_t, typename src_t>
__host__ __device__ inline typename std::enable_if<std::is_same<dst_t, src_t>::value, dst_t>::type
ggml_cuda_cast(src_t x) {
    return x;
}

template<typename dst_t, typename src_t>
__host__ __device__ inline typename std::enable_if<
    !std::is_same<dst_t, src_t>::value && std::is_same<dst_t, nv_bfloat16>::value, dst_t>::type
ggml_cuda_cast(src_t x) {
    return __float2bfloat16(float(x));
}

template<typename dst_t, typename src_t>
__host__ __device__ inline typename std::enable_if<
    !std::is_same<dst_t, src_t>::value && !std::is_same<dst_t, nv_bfloat16>::value &&
    std::is_same<src_t, nv_bfloat16>::value, dst_t>::type
ggml_cuda_cast(src_t x) {
    return __bfloat162float(x);
}

template<typename dst_t, typename src_t>
__host__ __device__ inline typename std::enable_if<
    std::is_same<src_t, float2>::value && std::is_same<dst_t, half2>::value, dst_t>::type
ggml_cuda_cast(src_t x) {
    return __float22half2_rn(x);
}

template<typename dst_t, typename src_t>
__host__ __device__ inline typename std::enable_if<
    std::is_same<src_t, nv_bfloat162>::value && std::is_same<dst_t, float2>::value, dst_t>::type
ggml_cuda_cast(src_t x) {
#ifdef GGML_USE_HIP
    return make_float2(__bfloat162float(__low2bfloat16(x)), __bfloat162float(__high2bfloat16(x)));
#else
#if __CUDA_ARCH__ >= 800
    return __bfloat1622float2(x);
#else
    return make_float2(__bfloat162float(x.x), __bfloat162float(x.y));
#endif // __CUDA_ARCH__ >= 800
#endif // GGML_USE_HIP
}

template<typename dst_t, typename src_t>
__host__ __device__ inline typename std::enable_if<
    std::is_same<src_t, float2>::value && std::is_same<dst_t, nv_bfloat162>::value, dst_t>::type
ggml_cuda_cast(src_t x) {
#ifdef GGML_USE_HIP
    return __float22bfloat162_rn(x);
#else
    return dst_t(nv_bfloat16(x.x), nv_bfloat16(x.y));
#endif // GGML_USE_HIP
}

template<typename dst_t, typename src_t>
__host__ __device__ inline typename std::enable_if<
    !std::is_same<dst_t, src_t>::value && !std::is_same<dst_t, nv_bfloat16>::value &&
    !std::is_same<src_t, nv_bfloat16>::value &&
    !(std::is_same<src_t, float2>::value && std::is_same<dst_t, half2>::value) &&
    !(std::is_same<src_t, nv_bfloat162>::value && std::is_same<dst_t, float2>::value) &&
    !(std::is_same<src_t, float2>::value && std::is_same<dst_t, nv_bfloat162>::value) &&
    std::is_same<dst_t, int32_t>::value, dst_t>::type
ggml_cuda_cast(src_t x) {
    return int32_t(x);
}

template<typename dst_t, typename src_t>
__host__ __device__ inline typename std::enable_if<
    !std::is_same<dst_t, src_t>::value && !std::is_same<dst_t, nv_bfloat16>::value &&
    !std::is_same<src_t, nv_bfloat16>::value &&
    !(std::is_same<src_t, float2>::value && std::is_same<dst_t, half2>::value) &&
    !(std::is_same<src_t, nv_bfloat162>::value && std::is_same<dst_t, float2>::value) &&
    !(std::is_same<src_t, float2>::value && std::is_same<dst_t, nv_bfloat162>::value) &&
    !std::is_same<dst_t, int32_t>::value, dst_t>::type
ggml_cuda_cast(src_t x) {
    return dst_t(float(x));
}
