

#include "test_main.h"
#include <iostream>
#ifdef _WIN32
#include <argparse.hpp>
#else
#endif
int main(int argc, char** argv)
{
    // test_ort_yolo8();
    //test_trt_yolo8();
    test_trt_yolo8seg();
    //test_classify();
    //build_trt();
    // test_trt_yolov5();
    //test_trt_ocr();
    // test_ort_ocr();
    // build_model(0, 1, "E:\\projdata\\ZXY_RPA\\best.onnx", "E:\\projdata\\ZXY_RPA\\best.trtmodel");
    //test_sync();
#ifdef _WIN32

    //argparse::ArgumentParser program("model_conversion_cmd");
    //program.add_argument("--onnx_model")
    //    .help(".onnx")
    //    .default_value(std::string(""));
    //program.add_argument("--trt_model")
    //    .help(".trtmodel")
    //    .default_value(std::string("./"));
    //try {
    //    program.parse_args(argc, argv);
    //}
    //catch (const std::runtime_error& err) {
    //    std::cerr << err.what() << std::endl;
    //    std::cerr << program;
    //    std::exit(1);
    //}
    //std::string onnx_model = program.get<std::string>("--onnx_model");
    //std::string trt_model = program.get<std::string>("--trt_model");
    //if (onnx_model != "") {
    //    //std::string tapp_path = "./models/";
    //    std::string onnx_path = trt_model + "/" + onnx_model + ".onnx";
    //    std::string trt_path = trt_model + "/" + onnx_model + ".trtmodel";
    //    // encode_onnx_model_to_binn(onnx_path.c_str(), trt_path.c_str());
    //    build_model(0, 6, onnx_path.c_str(), trt_path.c_str());
    //    std::cout << "conversion model succeesed!!!" << std::endl;
    //}
    //else {

    //    std::cout << "conversion model Fail。。。 " << std::endl;
    //}
#else
#endif
    GetAIRuntime()->DestoryRuntime();
    std::cout << "Hello World!" << std::endl;
    return 0;
}
