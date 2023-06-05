#include <cuda_runtime.h>
#include <cuda.h>

#define CUDA_CHECK(call)                              \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
        cudaGetErrorString(error_code));              \
        exit(1);                                      \
    }                                                 \
} while (0)

class TimeHelper {
public:
    static void init() {
        cudaEventCreate(&start_);
        cudaEventCreate(&end_);
    }

    static void TimeBegin() {
        cudaEventRecord(start_);
    }

    static void TimeBegin(cudaStream_t stream) {
        cudaEventRecord(start_, stream);
    }

    static void TimeEnd(float *time) {
        cudaEventRecord(end_);
        cudaEventSynchronize(end_);
        cudaEventElapsedTime(time, start_, end_);
    }

    static void TimeEnd(float *time, cudaStream_t stream) {
        cudaEventRecord(end_, stream);
        cudaEventSynchronize(end_);
        cudaEventElapsedTime(time, start_, end_);
    }

    static void destroy() {
        cudaEventDestroy(start_);
        cudaEventRecord(end_);
    }

private:
    static cudaEvent_t start_, end_;
};

#define MAX_CUDA(a, b)(a)>=(b)?(a):(b)
#define MIN_CUDA(a, b)(a)<=(b)?(a):(b)
