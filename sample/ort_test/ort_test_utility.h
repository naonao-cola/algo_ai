

#ifndef __ORT_TEST_UTILITY_H__
#define __ORT_TEST_UTILITY_H__

#include <fstream>
#include <mutex>
#include <string>

#include "opencv2/opencv.hpp"

#include "../../include/private/airuntime/AIRuntime.h"
#include "../../include/private/airuntime/inference.h"

#include "../../include/public/AIRuntimeDataStruct.h"
#include "../../include/public/AIRuntimeInterface.h"

cv::Mat draw_rst(cv::Mat image, Algo::BoxArray boxes);
cv::Mat draw_rst(cv::Mat image, std::vector<stResultItem> item_list);

#endif  // !__ORT_TEST_UTILITY_H__
