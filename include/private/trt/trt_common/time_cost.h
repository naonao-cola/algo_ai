#ifndef __TIME_COST_H__
#define __TIME_COST_H__

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

namespace TRT {
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
    bool is_stop_{false};
    high_resolution_clock::time_point start_time_;
    high_resolution_clock::time_point end_time_;
    milliseconds time_interval_;
};

} // TRT

#endif // __TIME_COST_H__