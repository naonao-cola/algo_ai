if is_os("linux") then
    set_languages("cxx17")
    add_cxxflags("-std=c++17")
    add_includedirs("/usr/include/spdlog/spdlog/include")

    add_linkdirs("../../../../../usr/lib")
    add_includedirs("../../../../../usr/local/include/opencv4")
    add_linkdirs("../../../../../usr/local/lib")

    add_links("opencv_gapi", "opencv_highgui", "opencv_ml", "opencv_objdetect", 
        "opencv_photo", "opencv_stitching", "opencv_video", "opencv_calib3d", 
        "opencv_features2d", "opencv_dnn", "opencv_flann", 
        "opencv_videoio", "opencv_imgcodecs", "opencv_imgproc", "opencv_core")
    add_links("nvinfer", "nvparsers", "nvinfer_plugin", "nvonnxparser", "nvparsers_static")

    target("AIFramework")
        set_kind("shared")
        add_includedirs("/usr/include/spdlog/spdlog/include")
        add_files("./**.cpp")
        add_files("./**.cu")
        --添加显示头文件
        add_headerfiles("../include/**.h")
        add_headerfiles("../include/**.hpp")
        add_headerfiles("../include/**.cuh")
        del_files("../include/private/ort/**")
    target_end()
elseif is_os("windows") then

    --自动更新vs解决方案结构
    add_rules("plugin.vsxmake.autoupdate")
    target("AIFramework")
        set_kind("shared")
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
        add_rules("rule_copy")
        add_rules("rule_display")
        -- 添加编译文件
        add_files("./**.cpp")
        add_files("./**.cu")
        --添加显示头文件
        add_headerfiles("../include/**.h")
        add_headerfiles("../include/**.hpp")
        add_headerfiles("../include/**.cuh")
    target_end()

else
    -- 其他操作系统
    print("Unsupported system")
end
