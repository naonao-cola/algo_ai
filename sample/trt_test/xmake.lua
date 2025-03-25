if is_os("linux") then

add_includedirs("/usr/include/spdlog/spdlog/include")
add_includedirs("../../../../../usr/local/include/opencv4")
-- add_linkdirs("../../../../../usr/local/lib")
-- add_linkdirs("../../../../..//usr/local/cuda/lib64")
-- add_requires("tensorRT")
-- add_packages("opencv")
add_links("opencv_gapi", "opencv_highgui", "opencv_ml", "opencv_objdetect", 
    "opencv_photo", "opencv_stitching", "opencv_video", "opencv_calib3d", 
    "opencv_features2d", "opencv_dnn", "opencv_flann", 
    "opencv_videoio", "opencv_imgcodecs", "opencv_imgproc", "opencv_core")
add_links("stdc++fs")
-- add_includedirs("../../../../..//usr/local/cuda/include")


target("trt_test")
    set_kind("binary")
    --添加三方库
    -- add_rules("package_cudnn")
    -- add_rules("package_tensorrt")
    add_rules("package_json")
    -- add_rules("package_format")
    -- add_rules("package_opencv")
    -- add_rules("package_spdlog")
    -- add_rules("package_queue")
    -- add_rules("package_cuda")
    -- add_rules("package_onnx")
    -- add_rules("rule_display")

    -- 添加编译文件
    add_files("./**.cpp")
    --add_files("./**.cu")
    --add_files("../../src/airuntime/*.cpp")
    add_files("../../src/trt/**.cpp")
    add_files("../../src/trt/**.cu")

    --添加显示头文件
    add_headerfiles("./*.h")
    add_headerfiles("../../include/public/*.h")
    add_headerfiles("../../include/private/airuntime/*.h")
    add_headerfiles("../../include/private/trt/**.h")
    add_headerfiles("../../include/private/trt/**.cuh")
    add_headerfiles("../../include/private/trt/**.hpp")
target_end()
elseif is_os("windows") then

target("trt_test")
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
    -- add_rules("package_onnx")
    add_rules("rule_display")
    -- 添加编译文件
    add_files("./**.cpp")
    --add_files("./**.cu")
    --add_files("../../src/airuntime/*.cpp")
    add_files("../../src/trt/**.cpp")
    add_files("../../src/trt/**.cu")

    --添加显示头文件
    add_headerfiles("./*.h")
    add_headerfiles("../../include/public/*.h")
    add_headerfiles("../../include/private/airuntime/*.h")
    add_headerfiles("../../include/private/trt/**.h")
    add_headerfiles("../../include/private/trt/**.cuh")
    add_headerfiles("../../include/private/trt/**.hpp")
target_end()
else
    -- 其他操作系统
    print("Unsupported operating system")
end
