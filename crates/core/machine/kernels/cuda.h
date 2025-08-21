#include <stdint.h>
#include <string.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename... Types> inline std::string fmt(const char* fmt, Types... args) {
  size_t len = std::snprintf(nullptr, 0, fmt, args...);
  std::string ret(++len, '\0');
  std::snprintf(&ret.front(), len, fmt, args...);
  ret.resize(--len);
  return ret;
}

#define CUDA_OK(expr)                                                                              \
  do {                                                                                             \
    cudaError_t code = expr;                                                                       \
    if (code != cudaSuccess) {                                            \
      auto msg = fmt("%s@%s:%d failed: \"%s\"",                                                    \
                     #expr,                                                                        \
                     __FILE__,                                                       \
                     __LINE__,                                                                     \
                     cudaGetErrorString(code));                                                    \
      throw std::runtime_error{msg};                                                               \
    }                                                                                              \
  } while (0)

class CudaStream {
private:
    cudaStream_t stream;

public:
    CudaStream() { cudaStreamCreate(&stream); }
    ~CudaStream() { cudaStreamDestroy(stream); }

    inline operator cudaStream_t() const { return stream; }
};

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared;

    LaunchConfig(dim3 grid, dim3 block, size_t shared = 0)
        : grid(grid), block(block), shared(shared) {}
    LaunchConfig(int grid, int block, size_t shared = 0) : grid(grid), block(block), shared(shared) {}
};

inline LaunchConfig getSimpleConfig(uint32_t count) {
    int device;
    CUDA_OK(cudaGetDevice(&device));

    int maxThreads;
    CUDA_OK(cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, device));

    int block = maxThreads / 4;
    int grid = (count + block - 1) / block;
    return LaunchConfig{grid, block, 0};
}

template <typename... ExpTypes, typename... ActTypes>
const char* launchKernel(void (*kernel)(ExpTypes...),
                         uint32_t count,
                         uint32_t shared_size,
                         ActTypes&&... args) {
    try {
        CudaStream stream;
        LaunchConfig cfg = getSimpleConfig(count);
        cudaLaunchConfig_t config;
        config.attrs = nullptr;
        config.numAttrs = 0;
        config.gridDim = cfg.grid;
        config.blockDim = cfg.block;
        config.dynamicSmemBytes = shared_size;
        config.stream = stream;
        CUDA_OK(cudaLaunchKernelEx(&config, kernel, std::forward<ActTypes>(args)...));
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const std::exception& err) {
        return strdup(err.what());
    } catch (...) {
        return strdup("Generic exception");
    }
    return nullptr;
}
