#ifndef __AIRUNTIMEUTILS_H__
#define __AIRUNTIMEUTILS_H__

#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using json = nlohmann::json;


inline json read_json_from_file(const char* json_path) {
    std::ifstream i(json_path);
    json j;
    i >> j;
    return j;
}

template<typename T>
inline T get_param(const json& param, const std::string& key, const T& def_val)
{
    if (param.contains(key)) {
        return param[key].get<T>();
    } else {
        return def_val;
    }
}

class TimeCost {
public:
    TimeCost() {
        is_stop_ = false;
    }

    void start() {
        start_time_ = high_resolution_clock::now();
    }

    void stop() {
        high_resolution_clock::time_point end_time_ = high_resolution_clock::now();
        time_interval_ = std::chrono::duration_cast<milliseconds>(end_time_ - start_time_);
        is_stop_ = true;
    }

    long long  get_cost_time() {
        if (is_stop_ == false) {
            stop();
        }
        return time_interval_.count();
    }

private:
    bool is_stop_{ false };
    high_resolution_clock::time_point start_time_;
    high_resolution_clock::time_point end_time_;
    milliseconds time_interval_;
};


#endif // __AIRUNTIMEUTILS_H__