set_project("trt_test")

set_version("0.0.1")
set_languages("c++17")

add_rules("mode.release","mode.releasedbg")

add_defines("USE_ORT=0") -- 1 启用  0不启用(ORT)
add_defines("USE_TRT=1") -- 1 启用  0不启用(TRT)


--自动更新vs解决方案结构
add_rules("plugin.vsxmake.autoupdate")


if is_mode "releasedbg" then
    -- set_symbols "hidden"
    -- set_optimize "fastest"
	-- set_runtimes("MT")
	--调试时打开下面两个
	set_optimize "none"
    set_symbols("debug")
end

--opencv 依赖库
if is_plat("windows") then
	add_syslinks("opengl32")
	add_syslinks("gdi32")
	add_syslinks("advapi32")
	add_syslinks("glu32")
	add_syslinks("ws2_32")
	add_syslinks("user32")
	add_syslinks("comdlg32")
end

target("tool_env")
    set_default(false)
    set_kind("phony")
    on_config(function (target)
        target:set("cudnn_dir", "E:/demo/rep/tv_ai_inference/3rdparty/cudnn-windows-x86_64-8.4.1.50_cuda11.6-archive/")
        target:set("tensorrt_dir", "E:/demo/rep/tv_ai_inference/3rdparty/TensorRT-8.6.1.6/")
        target:set("json_dir", "E:/demo/rep/tv_ai_inference/3rdparty/nlohmann-json_x64-windows/")
        target:set("format_dir", "E:/demo/rep/tv_ai_inference/3rdparty/fmt_x64-windows/")
        target:set("opencv_dir", "E:/demo/rep/tv_ai_inference/3rdparty/opencv/341/x64/")
        target:set("spd_dir", "E:/demo/rep/tv_ai_inference/3rdparty/spdlog_x64-windows/")
        target:set("queue_dir", "E:/demo/rep/tv_ai_inference/3rdparty/concurrent_queue/")
    end)
target_end()




-- cudnn库规则
rule("package_cudnn")
    on_config(function (target)
        import("core.project.project")
        local tool_env = project.target("tool_env")
        target:add("includedirs",path.join(tool_env:get("cudnn_dir"), "include"))
        target:add("linkdirs",path.join(tool_env:get("cudnn_dir"), "lib"))
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
        import("core.project.project")
        local tool_env = project.target("tool_env")
        target:add("includedirs",path.join(tool_env:get("tensorrt_dir"), "include"))
        target:add("linkdirs",path.join(tool_env:get("tensorrt_dir"), "lib"))
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
        import("core.project.project")
        local tool_env = project.target("tool_env")
        target:add("includedirs",path.join(tool_env:get("json_dir"), "include"))
    end)
rule_end()


--format库
rule("package_format")
    on_config(function (target)
        import("core.project.project")
        local tool_env = project.target("tool_env")
        target:add("includedirs",path.join(tool_env:get("format_dir"), "include"))
        target:add("linkdirs",path.join(tool_env:get("format_dir"), "lib"))
        target:add("links","fmt")

    end)
rule_end()

--opencv库
rule("package_opencv")
    on_config(function (target)
        import("core.project.project")
        local tool_env = project.target("tool_env")
        target:add("includedirs",path.join(tool_env:get("opencv_dir"), "include"))
        target:add("linkdirs",path.join(tool_env:get("opencv_dir"), "lib"))
        target:add("links","opencv_world341")

    end)
rule_end()

--spdlog库
rule("package_spdlog")
    on_config(function (target)
        import("core.project.project")
        local tool_env = project.target("tool_env")
        target:add("includedirs",path.join(tool_env:get("spd_dir"), "include"))
        target:add("linkdirs",path.join(tool_env:get("spd_dir"), "lib"))
        target:add("links","spdlog")

    end)
rule_end()

--队列库
rule("package_queue")
    on_config(function (target)
        import("core.project.project")
        local tool_env = project.target("tool_env")
        target:add("includedirs",tool_env:get("queue_dir"))
 end)
rule_end()

--添加cuda
rule("package_cuda")
    on_config(function (target)
        target:add("frameworks","cuda")
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
        os.cp("$(projectdir)/include/public/*.h","$(projectdir)/install")
    end)
rule_end()


target("01")
    add_deps("tool_env")
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
    add_rules("rule_display")
    -- 添加编译文件
    add_files("sample/**.cpp")
    add_files("src/trt/**.cpp")
    add_files("src/trt/**.cu")
    add_files("src/ort/**.cpp")

    --添加显示头文件
    add_headerfiles("sample/**.h")
    add_headerfiles("include/public/*.h")
    add_headerfiles("include/private/airuntime/*.h")
    add_headerfiles("include/private/trt/**.h")
    add_headerfiles("include/private/trt/**.cuh")
    add_headerfiles("include/private/trt/**.hpp")

target_end()

