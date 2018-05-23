/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 ******************************************************************************/

/******************************************************************************
 * Error handling utility routines
 ******************************************************************************/

#pragma once

#include <stdio.h>
#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {


/**
 * Displays error message in accordance with debug mode
 */
inline cudaError_t B40CPerror(
	cudaError_t error,
	const char *message,
	const char *filename,
	int line,
	bool print = true)
{
	if (error && print) {
		fprintf(stderr, "[%s, %d] %s (CUDA error %d: %s)\n", filename, line, message, error, cudaGetErrorString(error));
		fflush(stderr);
	}
	return error;
}

/**
 * Displays error message in accordance with debug mode
 */
inline cudaError_t B40CPerror(
	cudaError_t error,
	const char *filename,
	int line,
	bool print = true)
{
	if (error && print) {
		fprintf(stderr, "[%s, %d] (CUDA error %d: %s)\n", filename, line, error, cudaGetErrorString(error));
		fflush(stderr);
	}
	return error;
}

/**
 * Checks and resets last CUDA error.  If set, displays last error message in accordance with debug mode.
 */
inline cudaError_t B40CPerror(
	const char *message,
	const char *filename,
	int line,
	bool print = true)
{
	cudaError_t error = cudaGetLastError();
	if (error && print) {

		fprintf(stderr, "[%s, %d] %s (CUDA error %d: %s)\n", filename, line, message, error, cudaGetErrorString(error));
		fflush(stderr);
	}
	return error;
}

/**
 * Displays error message in accordance with debug mode
 */
inline cudaError_t B40CPerror(
	cudaError_t error,
	bool print = true)
{
	if (error && print) {
		fprintf(stderr, "(CUDA error %d: %s)\n", error, cudaGetErrorString(error));
		fflush(stderr);
	}
	return error;
}


/**
 * Checks and resets last CUDA error.  If set, displays last error message in accordance with debug mode.
 */
inline cudaError_t B40CPerror(
	bool print = true)
{
	cudaError_t error = cudaGetLastError();
	if (error && print) {
		fprintf(stderr, "(CUDA error %d: %s)\n", error, cudaGetErrorString(error));
		fflush(stderr);
	}
	return error;
}


} // namespace util
} // namespace b40c
B40C_NS_POSTFIX

