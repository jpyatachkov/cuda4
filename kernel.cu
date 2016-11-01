
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <stdio.h>
#include <vector>

namespace Device {
	float *data   = nullptr;
	float *result = nullptr;

	const auto blockSize = 3;
}

namespace HostConstants {
	const std::size_t inFilenamePos  = 1;
	const std::size_t outFilenamePos = 2;

	const char *delimiter         = " ";

	const std::size_t bufferSize  = 500;
}

/*
* CUDA errors catching block
*/

static void _checkCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define cudaCheck(value) _checkCudaErrorAux(__FILE__, __LINE__, #value, value)

static void _checkCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;

	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;

	_gpuFree();

	system("pause");

	exit(1);
}

/*
 * GPU helpers
 */

static void _gpuInit(std::size_t size) {
	cudaCheck(cudaMalloc(&Device::data, size * sizeof(*Device::data)));
	cudaCheck(cudaMalloc(&Device::result, size * sizeof(*Device::result)));
}

static void _gpuFree() {
	if (Device::data != nullptr)
		cudaCheck(cudaFree((void *)Device::data));

	if (Device::result != nullptr)
		cudaCheck(cudaFree((void *)Device::result));
}

/*
 * Kernel
 */

__global__ void borderDetectorKernel(const float *data, float *result, const std::size_t w, const std::size_t h) {
		auto xIdx = threadIdx.x + blockIdx.x * blockDim.x;
		auto yIdx = threadIdx.y + blockIdx.y * blockDim.y;

		const auto G_L = 150;
		const auto G_H = 200;

		if (xIdx < w && yIdx < h) {
			auto gX =     data[xIdx + yIdx * w]           -     data[xIdx + 2 + yIdx * w] +
					  2 * data[xIdx + (yIdx + 1) * w]     - 2 * data[xIdx + 2 + (yIdx + 2) * w] +
						  data[xIdx + (yIdx + 2) * w]     -     data[xIdx + 2 + (yIdx * 2) * w];

			auto gY =    data[xIdx + yIdx * w]       + 2 * data[xIdx + 1 + yIdx * w]       + data[xIdx + 2 + yIdx * w]
					   - data[xIdx + (yIdx + 2) * w] - 2 * data[xIdx + 1 + (yIdx + 2) * w] - data[xIdx + 2 + (yIdx + 2) * w];

			auto g  = sqrtf(gX * gX + gY * gY);

			result[xIdx + yIdx * w] = (g > G_L && g < G_H) ? 0 : 255;
		}
}

// TODO: shared kernel

/*
 * Helpers
 */

 std::vector<std::string> split(const std::string &stringToSplit, const std::string &delimiter = " ") {
     std::regex re(delimiter);

     return {std::sregex_token_iterator(stringToSplit.begin(), stringToSplit.end(), re, -1), std::sregex_token_iterator()};
 }

/*
 * Main
 */

int main(int argc, const char *argv[]) {
	// TODO: out file

	using namespace HostConstants;

  if (argc != 3) {
      std::cout << "Задано неверное количество аргументов" << std::endl;
      return 1;
  }

  std::ifstream ifs(argv[inFilenamePos], std::ios_base::in);

  if (!ifs.is_open()) {
      std::cout << "Невозможно открыть файл " << argv[inFilenamePos] << std::endl;
      return 1;
  }

  std::vector<float> valuesFromFile;

  std::size_t w = 0, h = 0;

  try {
      std::string line;

      while (!ifs.eof()) {
          std::getline(ifs, line);

          if (line.size() == 0)
              continue;

          auto values = split(line, delimiter);

          for (auto &value : values)
              valuesFromFile.push_back(std::stoi(value));

          if (w != 0 && values.size() != w)
              throw std::invalid_argument("Строки в файле содержат разное количество пикселов");

          if (w == 0)
              w = valuesFromFile.size();
          h++;
      }
  }
  catch (std::invalid_argument &ia) {
      std::cout << ia.what() << std::endl;
      return 1;
  }
  catch (...) {
      std::cout << "При чтении из файла произошла ошибка" << std::endl;
      return 1;
  }

  const auto size = valuesFromFile.size();

	_gpuInit(size);

	cudaCheck(cudaMemcpy(Device::data, valuesFromFile.data(), size * sizeof(valuesFromFile[0]), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemset(Device::result, 0, size * sizeof(*Device::result)));

	auto nThreads = 63;
	
	auto nBlocksX = (w % nThreads == 0) ? w / nThreads : w / nThreads + 1;
	auto nBlocksY = (h % nThreads == 0) ? h / nThreads : h / nThreads + 1;

	dim3 nBlocks(nBlocksX, nBlocksY);
	dim3 nThreads(nThreads, nThreads);

	borderDetectorKernel <<<nBlocks, nThreads>>> (Device::data, Device::result, width, height);

	cudaCheck(cudaMemcpy(valuesFromFile.data(), Device::data, size * sizeof(*Device::data), cudaMemcpyDeviceToHost));

	for (auto &item : valuesFromFile)
		std::cout << item << " ";

	return 0;
}
