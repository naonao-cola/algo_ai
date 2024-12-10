#pragma once
// #include <Windows.h>
#ifdef _WIN32
    #include <windows.h>
#elif __linux__
#endif
#include <chrono>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <future>

using TimePoint = std::chrono::system_clock::time_point;

struct tTaktTime
{
	TimePoint startTime;
	TimePoint endTime;
	long long costTimeMs;
};

struct tAlgoResultItem
{
	std::string code;
};

struct tAlgoResults
{
	tTaktTime tt;
	std::vector<std::shared_ptr<tAlgoResultItem>> vecResults;
};



struct tAlgoInspParam
{
	int nImageNum;
	int nROINum;
	int nAlgNum;
	bool bpInspectEnd;
	 HANDLE hInspectEnd;
	tAlgoResults algoResults;
	cv::Mat imgdata;
};


