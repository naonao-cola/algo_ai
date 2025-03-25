if is_os("linux") then
add_links("stdc++fs")
target("test_so")
    set_kind("binary")
    add_rules("rule_display")
    add_rules("package_json")
    add_deps("AIFramework")
    add_files("./**.cpp")
    add_headerfiles("../../install/*.h")
elseif is_os("windows") then
target("test_dll")
    set_kind("binary")
    set_default("test_dll")
    add_rules("rule_display")
    add_rules("package_opencv")
    add_rules("package_json")
    add_rules("package_argparse")
    add_deps("AIFramework")
    add_files("./**.cpp")
    add_headerfiles("./*.h")
    add_headerfiles("../../install/*.h")
else
    print("Unsupported system")
end
