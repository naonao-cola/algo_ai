#pragma once
// #include <Windows.h>
#ifdef _WIN32
    #include <windows.h>
#elif __linux__
#endif
#include <iostream>
#include <vector>
using namespace std;

#ifdef _WIN32
    
#elif __linux__
#define HANDLE void*
#endif
// #define USE_MPMC

#define DEF_PRE_THRD_CNT 4
#define DEF_POST_THRD_CNT 2
#define DEF_PRE_THRD_PRIORITY   THREAD_PRIORITY_NORMAL
#define DEF_POST_THRD_PRIORITY  THREAD_PRIORITY_ABOVE_NORMAL
#define PREP_QUEUE_MAX_SIZE 128
#define POST_QUEUE_MAX_SIZE 128
#define WAIT_DEQUEUE_TIMEOUT 1000

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


