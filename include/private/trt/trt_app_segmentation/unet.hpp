#pragma once

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

#include "../../airuntime/inference.h"

namespace UNet {
	std::shared_ptr<Algo::Infer> create_infer(const std::string& engine_file, int gpuid,float confidence_threshold = 0.25f
);
} // namespace UNet
