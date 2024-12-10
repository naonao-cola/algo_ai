#ifndef __ORT_INFER_H__
#define __ORT_INFER_H__
#if USE_ORT

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
#include "tensorrt_provider_factory.h"

namespace ort {

class BasicOrtInfer
{
public:
    explicit BasicOrtInfer(const std::string& onnx_path, unsigned int num_threads = 8, bool use_gpu = false);
    ~BasicOrtInfer();
    std::string                       get_onnx_path() const;
    Ort::Session*                     get_session() const;
    std::vector<std::vector<int64_t>> get_input_node_dims() const;
    std::vector<std::vector<int64_t>> get_out_node_dims() const;

    std::vector<const char*> get_input_node_names() const;
    std::vector<const char*> get_output_node_names() const;

    void             print_debug_string();
    BasicOrtInfer*   get_engine();
    Ort::MemoryInfo& get_memory_info();
    size_t           get_num_outputs() const;
    size_t           get_num_intputs() const;

private:
    const char*     _onnx_path = nullptr;
    Ort::Env        _ort_env;
    Ort::Session*   _ort_session         = nullptr;
    Ort::MemoryInfo _memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // input
    size_t                            _num_inputs = 1;
    std::vector<const char*>          _input_node_names;
    std::vector<std::vector<int64_t>> _input_node_dims;

    // output
    size_t                            _num_outputs = 1;
    std::vector<const char*>          _output_node_names;
    std::vector<std::vector<int64_t>> _output_node_dims;

    const char*  _log_id      = nullptr;
    bool         _use_gpu     = true;
    unsigned int _num_threads = 8;  // initialize at runtime.

    void                     initialize_handler();
    std::vector<std::string> _output_node_str_names;
    std::vector<std::string> _input_node_str_names;
};
}  // namespace ort
#endif

#endif //USE_ORT