#ifdef _WIN32

#if USE_ORT


#include "../../../include/private/ort/ort_common/ort_infer.hpp"
#include "../../../include/private/ort/ort_common/ort_utility.h"

namespace ort {
BasicOrtInfer::BasicOrtInfer(const std::string& onnx_path, unsigned int num_threads, bool use_gpu)
    : _log_id(onnx_path.data()), _use_gpu(use_gpu), _num_threads(num_threads)
{
    _onnx_path = onnx_path.data();
    initialize_handler();
}

void BasicOrtInfer::initialize_handler()
{
    _ort_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, _log_id);
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(_num_threads);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetLogSeverityLevel(4);

    if (_use_gpu) {
        OrtStatusPtr cuda_ptr = OrtSessionOptionsAppendExecutionProvider_CUDA(
            session_options, 0);  // C API stable.
        if (cuda_ptr != NULL) {
            std::cout << "onnxruntime GPU engine false "
                      << "\n";
        }
    }

    _ort_session = new Ort::Session(
        _ort_env, ort::utils::to_wstring(_onnx_path).c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    _num_inputs = _ort_session->GetInputCount();
    _input_node_names.resize(_num_inputs);
    for (unsigned int i = 0; i < _num_inputs; ++i) {
        auto        input_name = _ort_session->GetInputNameAllocated(i, allocator);
        std::string str_tmp    = input_name.get();
        _input_node_str_names.emplace_back(str_tmp);
        Ort::TypeInfo input_type_info   = _ort_session->GetInputTypeInfo(i);
        auto          input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto          input_dims        = input_tensor_info.GetShape();
        _input_node_dims.push_back(input_dims);
    }
    for (unsigned int i = 0; i < _num_inputs; ++i) {
        _input_node_names[i] = _input_node_str_names[i].c_str();
    }

    _num_outputs = _ort_session->GetOutputCount();
    _output_node_names.resize(_num_outputs);
    for (unsigned int i = 0; i < _num_outputs; ++i) {
        auto        output_name = _ort_session->GetOutputNameAllocated(i, allocator);
        std::string str_tmp     = output_name.get();
        _output_node_str_names.emplace_back(str_tmp);
        Ort::TypeInfo output_type_info   = _ort_session->GetOutputTypeInfo(i);
        auto          output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto          output_dims        = output_tensor_info.GetShape();
        _output_node_dims.push_back(output_dims);
    }
    for (unsigned int i = 0; i < _num_outputs; i++) {
        _output_node_names[i] = _output_node_str_names[i].c_str();
    }
}

void BasicOrtInfer::print_debug_string()
{
    std::cout << "LITEORT_DEBUG LogId: " << _onnx_path << "\n";
    std::cout << "=============== Input-Dims ==============\n";
    for (unsigned int i = 0; i < _num_inputs; i++) {
        for (unsigned int j = 0; j < _input_node_dims.at(i).size(); ++j) {
            std::cout << "Input: " << i << "  Name: " << _input_node_names.at(i)
                      << " Dim: " << j << " :" << _input_node_dims.at(i).at(j)
                      << "\n";
        }
    }

    std::cout << "=============== Output-Dims ==============\n";
    for (unsigned int i = 0; i < _num_outputs; ++i) {
        for (unsigned int j = 0; j < _output_node_dims.at(i).size(); ++j) {
            std::cout << "Output: " << i << " Name: " << _output_node_names.at(i)
                      << " Dim: " << j << " :" << _output_node_dims.at(i).at(j)
                      << std::endl;
        }
    }

    std::cout << "========================================\n";
}

BasicOrtInfer::~BasicOrtInfer()
{
    if (_ort_session)
        delete _ort_session;
    _ort_session = nullptr;
}
std::string                       BasicOrtInfer::get_onnx_path() const { return this->_onnx_path; }
Ort::Session*                     BasicOrtInfer::get_session() const { return this->_ort_session; }
std::vector<std::vector<int64_t>> BasicOrtInfer::get_input_node_dims() const
{
    return this->_input_node_dims;
}
std::vector<std::vector<int64_t>> BasicOrtInfer::get_out_node_dims() const
{
    return this->_output_node_dims;
}

std::vector<const char*> BasicOrtInfer::get_input_node_names() const
{
    return this->_input_node_names;
}
std::vector<const char*> BasicOrtInfer::get_output_node_names() const
{
    return this->_output_node_names;
}
BasicOrtInfer*   BasicOrtInfer::get_engine() { return this; }
Ort::MemoryInfo& BasicOrtInfer::get_memory_info()
{
    return this->_memory_info_handler;
}
size_t BasicOrtInfer::get_num_outputs() const { return this->_num_outputs; }
size_t BasicOrtInfer::get_num_intputs() const { return this->_num_inputs; }
}  // namespace ort


#endif //USE_ORT
#elif __linux__
#endif