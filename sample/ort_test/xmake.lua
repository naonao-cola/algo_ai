
-- target("ort_test")
--     set_kind("binary")
--     --添加三方库
--     -- add_rules("package_cudnn")
--     -- add_rules("package_tensorrt")
--     -- add_rules("package_json")
--     -- add_rules("package_format")
--     -- add_rules("package_opencv")
--     -- add_rules("package_spdlog")
--     -- add_rules("package_queue")
--     -- add_rules("package_cuda")
--     -- add_rules("package_onnx")
--     -- add_rules("rule_display")
--     -- 添加编译文件,ort的部分不编译 trt的部分
--     add_files("./**.cpp")
--     add_files("../../src/trt/trt_app_ocr/ocr_utility.cpp")
--     add_files("../../src/trt/trt_app_ocr/clipper.cpp")

--     --add_files("./**.cu")
--     --add_files("../../src/airuntime/*.cpp")
--     add_files("../../src/ort/**.cpp")
--     --add_files("../../src/ort/**.cu")
--     --添加显示头文件
--     add_headerfiles("./*.h")
--     add_headerfiles("../../include/public/*.h")
--     add_headerfiles("../../include/private/airuntime/*.h")
--     add_headerfiles("../../include/private/ort/**.h")
--     add_headerfiles("../../include/private/ort/**.cuh")
--     add_headerfiles("../../include/private/ort/**.hpp")
-- target_end()
-- else
if is_os("windows") then
target("ort_test")
    set_kind("binary")
    --添加三方库
    add_rules("package_cudnn")
    add_rules("package_tensorrt")
    add_rules("package_json")
    add_rules("package_format")
    add_rules("package_opencv")
    add_rules("package_spdlog")
    add_rules("package_queue")
    add_rules("package_cuda")
    add_rules("package_onnx")
    add_rules("rule_display")
    -- 添加编译文件,ort的部分不编译 trt的部分
    add_files("./**.cpp")
    add_files("../../src/trt/trt_app_ocr/ocr_utility.cpp")
    add_files("../../src/trt/trt_app_ocr/clipper.cpp")

    --add_files("../../src/airuntime/*.cpp")
    add_files("../../src/ort/**.cpp")
    --添加显示头文件
    add_headerfiles("./*.h")
    add_headerfiles("../../include/public/*.h")
    add_headerfiles("../../include/private/airuntime/*.h")
    add_headerfiles("../../include/private/ort/**.h")
    add_headerfiles("../../include/private/ort/**.hpp")
target_end()
else
    -- 其他操作系统
    print("Unsupported operating system")
end
