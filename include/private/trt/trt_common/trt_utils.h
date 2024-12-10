#ifndef __TRT_UTILS_H__
#define __TRT_UTILS_H__
#if USE_TRT


#include <NvInfer.h>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include "cuda_runtime.h"

namespace TRT {

	template<typename _T> std::shared_ptr<_T> make_nvshared(_T* ptr);

	std::vector<unsigned char> load_file(const std::string& file);
} // TRT

#endif // __TRT_UTILS_H__


#endif //USE_TRT