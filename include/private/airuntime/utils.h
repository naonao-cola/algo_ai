#ifndef UTILS
#define UTILS
#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;

template<typename T>
inline T get_param(const json& param, const std::string& key, const T& def_val)
{
    if (param.contains(key)) {
        return param[key].get<T>();
    } else {
        return def_val;
    }
}

#endif // !UTILS