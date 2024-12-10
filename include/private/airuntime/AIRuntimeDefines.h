#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "fmt/format.h"


#ifndef DECLARE_SINGLETON
#define DECLARE_SINGLETON(class_name) \
    static class_name* get_instance() { \
        static class_name instance; \
        return &instance; \
    }
#endif // !DECLARE_SINGLETON

#ifndef HIDE_CREATE_METHODS
#define HIDE_CREATE_METHODS(class_name) \
    class_name(); \
    ~class_name(); \
    class_name(const class_name&) = delete; \
    class_name& operator=(const class_name&) = delete; 
#endif // !HIDE_CREATE_METHODS
