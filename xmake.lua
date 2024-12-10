add_defines("USE_ORT=1") -- 1 启用  0不启用(ORT)
add_defines("USE_TRT=1") -- 1 启用  0不启用(TRT)

if is_os("linux") then
set_project("AIRuntime")
    -- Linux系统的编译配置
set_version("0.0.1")
set_languages("c++17")
set_languages("cxx17")
    -- 添加编译标志
add_cxflags("-allow-unsupported-compiler")
add_rules("mode.debug", "mode.release")
add_links("stdc++fs")
if is_mode "release" then
    set_symbols "hidden"
    set_optimize "fastest"
	set_runtimes("MT")
	--调试时打开下面两个
	-- set_optimize "none"
    -- set_symbols("debug")
end

--json库
rule("package_json")
    on_config(function (target)
        target:add("includedirs","3rdparty/nlohmann-json_x64-windows/include")
    end)
rule_end()

--队列库
rule("package_queue")
    on_config(function (target)
    target:add("includedirs","3rdparty/concurrent_queue")
 end)
rule_end()

--显示构建目标路径
rule("rule_display")
     after_build(function (target)
     cprint("${green} BIUD TARGET: %s", target:targetfile())
    end)
rule_end()



--构建完成后复制文件
rule("rule_copy")
    after_build(function (target)
        os.cp(target:targetfile(), "$(projectdir)/install")
        --os.rm(target:targetfile())
        os.cp("$(projectdir)/include/public/*.h","$(projectdir)/install")
    end)
rule_end()

--自动更新vs解决方案结构
add_rules("plugin.vsxmake.autoupdate")
add_includedirs("../../../../../usr/local/include/opencv4")
add_linkdirs("../../../../../usr/local/lib")
add_links("opencv_gapi", "opencv_highgui", "opencv_ml", "opencv_objdetect", 
    "opencv_photo", "opencv_stitching", "opencv_video", "opencv_calib3d", 
    "opencv_features2d", "opencv_dnn", "opencv_flann", 
    "opencv_videoio", "opencv_imgcodecs", "opencv_imgproc", "opencv_core")
add_links("nvinfer", "nvparsers", "nvinfer_plugin", "nvonnxparser", "nvparsers_static")
add_includedirs("../../../../../usr/local/cuda-11.7/include")
add_linkdirs("../../../../../usr/local/cuda-11.7/lib64")

includes(
    "src/xmake.lua",
    -- "sample/ort_test/xmake.lua",
    "sample/trt_test/xmake.lua",
    "sample/dll_test/xmake.lua"
)

elseif is_os("windows") then

set_project("AIRuntime")
if is_os("linux") then
    print("Operating System: Linux")
    print("Architecture: " .. (is_arch("x86_64") and "x86_64" or "Other"))


    -- 添加特定的编译选项或依赖
elseif is_os("windows") then
end
set_version("0.0.1")
set_languages("c++17")
add_rules("mode.debug", "mode.release")

if is_mode "release" then
    -- set_symbols "hidden"
    -- set_optimize "fastest"
	-- set_runtimes("MT")
	--调试时打开下面两个
	set_optimize "none"
    set_symbols("debug")
end

-- cudnn库规则
rule("package_cudnn")
    on_config(function (target)
        target:add("includedirs","3rdparty/cudnn-windows-x86_64-8.4.1.50_cuda11.6-archive/include")
        target:add("linkdirs","3rdparty/cudnn-windows-x86_64-8.4.1.50_cuda11.6-archive/lib")
        target:add("links",
        "cudnn",
        "cudnn64_8",
        "cudnn_adv_infer",
        "cudnn_adv_infer64_8",
        "cudnn_adv_train",
        "cudnn_adv_train64_8",
        "cudnn_cnn_infer",
        "cudnn_cnn_infer64_8",
        "cudnn_cnn_train",
        "cudnn_cnn_train64_8",
        "cudnn_ops_infer",
        "cudnn_ops_infer64_8",
        "cudnn_ops_train",
        "cudnn_ops_train64_8"
        )
    end)
rule_end()

-- tensorrt库
rule("package_tensorrt")
    on_config(function (target)
        target:add("includedirs","3rdparty/TensorRT-8.4.1.5/include")
        target:add("linkdirs","3rdparty/TensorRT-8.4.1.5/lib")
        target:add("links",
        "nvinfer",
        "nvonnxparser",
        "nvinfer_plugin",
        "nvparsers"
        )
    end)
rule_end()

--json库
rule("package_json")
    on_config(function (target)
        target:add("includedirs","3rdparty/nlohmann-json_x64-windows/include")
    end)
rule_end()

--argparse库
rule("package_argparse")
    on_config(function (target)
        target:add("includedirs","3rdparty/argparse")
    end)
rule_end()

--format库
rule("package_format")
    on_config(function (target)
        target:add("includedirs","3rdparty/fmt_x64-windows/include")
        if is_mode("release") then
            target:add("linkdirs","3rdparty/fmt_x64-windows/lib")
            target:add("links","fmt")
        else
            target:add("linkdirs","3rdparty/fmt_x64-windows/debug/lib")
            target:add("links","fmtd")
        end
    end)
rule_end()

--opencv库
rule("package_opencv")
    on_config(function (target)
        target:add("includedirs","3rdparty/opencv/341/x64/include")
        target:add("linkdirs","3rdparty/opencv/341/x64/lib")
        if is_mode("release") then
            target:add("links","opencv_world341")
        else
            target:add("links","opencv_world341d")
        end
    end)
rule_end()

--spdlog库
rule("package_spdlog")
    on_config(function (target)
        target:add("includedirs","3rdparty/spdlog_x64-windows/include")
        if is_mode("release") then
            target:add("linkdirs","3rdparty/spdlog_x64-windows/lib")
            target:add("links","spdlog")
        else
            target:add("linkdirs","3rdparty/spdlog_x64-windows/debug/lib")
            target:add("links","spdlogd")
        end
    end)
rule_end()

--队列库
rule("package_queue")
    on_config(function (target)
    target:add("includedirs","3rdparty/concurrent_queue")
 end)
rule_end()

--添加cuda
rule("package_cuda")
    on_config(function (target)
        target:add("frameworks","cuda")
    end)
rule_end()

rule("package_onnx")
    on_config(function (target)
        target:add("includedirs","3rdparty/onnxruntime-win-x64-gpu-1.15.1/include")
        target:add("linkdirs","3rdparty/onnxruntime-win-x64-gpu-1.15.1/lib")
        target:add("links",
        "onnxruntime",
        "onnxruntime_providers_cuda",
        "onnxruntime_providers_shared",
        "onnxruntime_providers_tensorrt"
        )
    end)
rule_end()



--显示构建目标路径
rule("rule_display")
     after_build(function (target)
     cprint("${green}  BIUD TARGET: %s", target:targetfile())
    end)
rule_end()



--构建完成后复制文件
rule("rule_copy")
    after_build(function (target)
        os.cp(target:targetfile(), "$(projectdir)/install")
        --os.rm(target:targetfile())
        os.cp("$(projectdir)/include/public/*.h","$(projectdir)/install")
    end)
rule_end()

--自动更新vs解决方案结构
add_rules("plugin.vsxmake.autoupdate")

includes(
    "src/xmake.lua",
    "sample/ort_test/xmake.lua",
    -- "sample/trt_test/xmake.lua",
    "sample/dll_test/xmake.lua"
)

else
    -- 其他操作系统
    print("Unsupported operating system")
end



