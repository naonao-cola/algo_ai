#ifndef __CLASSIFICATION__HPP__
#define __CLASSIFICATION__HPP__


#include <memory>
#include "../../airuntime/inference.h"


namespace Classification {
std::shared_ptr<Algo::Infer> create_infer(const std::string& engine_file, int gpuid, float confidence_threshold = 0.25f, std::vector<std::vector<int>> dims = {});
} //namespace Classification 


#endif // !__CLASSIFICATION__HPP__
