#ifndef TRT_BUILDER_HPP
#define TRT_BUILDER_HPP

#include <string>
#include <vector>
#include <functional>

namespace TRT {

	/**
	 * @brief Model Type.
	 */
	enum class Mode : int {
		FP32,
		FP16
	};

	/**
	 * @brief Compile onnx to tensorRT engine
	 * mode:				FP32 | FP16
	 * source:				onnx path
	 * saveto				tensorRT engine file save path
	 * maxWorkspaceSize		maxWorkspaceSize
	 */
	bool compile(Mode mode,unsigned int maxBatchSize,const std::string& source,const std::string& saveto,const size_t maxWorkspaceSize = 1ul << 30 );
};

#endif //TRT_BUILDER_HPP