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
	int *data   = nullptr;
	int *result = nullptr;
}

namespace HostConstants {
	const std::size_t inFilenamePos  = 1;
	const std::size_t outFilenamePos = 2;

	const char *delimiter         = " ";

	const std::size_t bufferSize  = 500;
}

static void _gpuFree();

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

static void _gpuInit(std::size_t sizeData, std::size_t sizeResult) {
	cudaCheck(cudaMalloc(&Device::data, sizeData * sizeof(*Device::data)));
	cudaCheck(cudaMalloc(&Device::result, sizeResult * sizeof(*Device::result)));
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

__global__ void smootherKernel(int *data, const std::size_t w, const std::size_t h) {
	int G[]    = { 2, 4,  5,  4,  2,
		           4, 9,  12, 9,  4,
		           5, 12, 15, 12, 5,
		           4, 9,  12, 9,  4,
		           2, 4,  5,  4,  2 };

	const auto size      = 5;
	const auto nElements = size * size - 1;

	auto xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	auto yIdx = threadIdx.y + blockIdx.y * blockDim.y;

	if (xIdx < h && yIdx < w) {
		auto result = 0;

		for (auto i = 0; i < size; i++) {
			for (auto j = 0; j < size; j++)
				result += data[(xIdx + i) * w + yIdx + j] * G[nElements - (i * size + j)];
		}

		data[xIdx * w + yIdx] = result / 159;
	}
}

__global__ void borderDetectorKernel(int *data, int *result, const std::size_t w, const std::size_t h) {
	auto xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	auto yIdx = threadIdx.y + blockIdx.y * blockDim.y;

	const auto G_L = 150;
	const auto G_H = 200;

	if (xIdx < h && yIdx < w) {
		auto gX =   + data[xIdx * w + yIdx]           -    data[xIdx * w + yIdx + 2] +
				  2 * data[(xIdx + 1) * w + yIdx]     - 2 * data[(xIdx + 1) * w + yIdx + 2] +
					  data[(xIdx + 2) * w + yIdx]     -     data[(xIdx + 2) * w + yIdx + 2];

		auto gY =    data[xIdx * w + yIdx]       + 2 * data[xIdx * w + yIdx + 1]       + data[xIdx * w + yIdx + 2]
			       - data[(xIdx + 2) * w + yIdx] - 2 * data[(xIdx + 2) * w + yIdx + 1] - data[(xIdx + 2) * w + yIdx + 2];

		auto g  = __fsqrt_ru(gX * gX + gY * gY);

		result[xIdx * w + yIdx] = (g > G_L && g < G_H) ? 255 : 0;
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
	// TODO: out to file

	// 1 аргумент - путь ко входному файлу
	// 2 аргумент - путь к выходному файлу

	using namespace HostConstants;

	if (argc != 3) {
		std::cout << "Задано неверное количество аргументов" << std::endl;
		return 1;
	}

	std::ifstream ifs(argv[inFilenamePos], std::ios_base::in);
	std::ofstream ofs(argv[outFilenamePos], std::ios_base::out | std::ios_base::trunc);

	if (!ifs.is_open()) {
		std::cout << "Невозможно открыть файл " << argv[inFilenamePos] << std::endl;
		system("pause");
		return 1;
	}

	if (!ofs.is_open()) {
		std::cout << "Невозможно открыть файл " << argv[outFilenamePos] << std::endl;
		system("pause");
		return 1;
	}

	std::vector<int> valuesFromFile;

	auto w = 960, h = 512;

	try {
		int value = 0;

		while (!ifs.eof()) {
			ifs >> value;
			valuesFromFile.push_back(value);
		}

		// for correct operating of last 4 values in each column of image matrix
		std::vector<int> zeroVec(w + 4, 0);

		for (auto i = 0; i < 4; i++)
			valuesFromFile.insert(valuesFromFile.end(), zeroVec.begin(), zeroVec.end());

		ifs.close();
	}
	catch (std::invalid_argument &ia) {
		std::cout << ia.what() << std::endl;
		system("pause");
		return 1;
	}
	catch (...) {
		std::cout << "При чтении из файла произошла ошибка" << std::endl;
		system("pause");
		return 1;
	}

	const auto sizeData   = valuesFromFile.size();
	const auto sizeResult = w * h;

	_gpuInit(sizeData, sizeResult);

	cudaCheck(cudaMemcpy(Device::data, valuesFromFile.data(), sizeData * sizeof(valuesFromFile[0]), cudaMemcpyHostToDevice));

	cudaCheck(cudaMemset(Device::result, 0, sizeResult * sizeof(*Device::result)));

	auto threads = 32;
	
	auto nBlocksX = (h % threads == 0) ? h / threads : h / threads + 1;
	auto nBlocksY = (w % threads == 0) ? w / threads : w / threads + 1;

	dim3 nBlocks(nBlocksX, nBlocksY);
	dim3 nThreads(threads, threads);

	smootherKernel <<<nBlocks, nThreads>>> (Device::data, w, h);
	borderDetectorKernel <<<nBlocks, nThreads>>> (Device::data, Device::result, w, h);

	valuesFromFile.resize(sizeResult);

	cudaCheck(cudaMemcpy(valuesFromFile.data(), Device::result, sizeResult * sizeof(*Device::result), cudaMemcpyDeviceToHost));

	auto elementsInLine = 0;

	for (auto &item : valuesFromFile) {
		ofs << item << " ";

		if (++elementsInLine == w) {
			elementsInLine = 0;
			ofs << std::endl;
		}
	}

	_gpuFree();

	system("pause");

	return 0;
}
