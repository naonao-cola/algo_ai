#ifndef CUDA_TOOLS_HPP
#define CUDA_TOOLS_HPP
#if USE_TRT


#include <cuda.h>
#include <cuda_runtime.h>
#include "./ilogger.hpp"

#define GPU_BLOCK_THREADS  512


#define KernelPositionBlock											\
	int position = (blockDim.x * blockIdx.x + threadIdx.x);		    \
    if (position >= (edge)) return;


// #define checkCudaDriver(call)  CUDATools::check_driver(call, #call, __LINE__, __FILE__)
#define checkCudaRuntime(call) CUDATools::checkRuntime(call, #call, __LINE__, __FILE__)

#define checkCudaKernel(...)                                                                         \
    __VA_ARGS__;                                                                                     \
    do{cudaError_t cudaStatus = cudaPeekAtLastError();                                               \
    if (cudaStatus != cudaSuccess){                                                                  \
        INFOE("launch failed: %s", cudaGetErrorString(cudaStatus));                                  \
    }} while(0);


#define Assert(op)					 \
	do{                              \
		bool cond = !(!(op));        \
		if(!cond){                   \
			INFOF("Assert failed, " #op);  \
		}                                  \
	}while(false)


struct CUctx_st;
struct CUstream_st;

using ICUStream = CUstream_st *;
using ICUContext = CUctx_st *;
using ICUDeviceptr = void *;
using DeviceID = int;

namespace CUDATools{
    bool check_driver(CUresult e, const char* call, int iLine, const char *szFile);
    bool checkRuntime(cudaError_t e, const char* call, int iLine, const char *szFile);
    bool check_device_id(int device_id);
    int current_device_id();

    dim3 grid_dims(int numJobs);
    dim3 block_dims(int numJobs);

    // return 8.6  etc.
    std::string device_capability(int device_id);
    std::string device_name(int device_id);
    std::string device_description();

    class AutoDevice{
    public:
        AutoDevice(int device_id = 0);
        virtual ~AutoDevice();
    
    private:
        int old_ = -1;
    };
}

#endif // CUDA_TOOLS_HPP
#endif //USE_TRT