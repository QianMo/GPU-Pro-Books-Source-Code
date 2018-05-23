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
 * Simple test driver program for radix sort.
 ******************************************************************************/

#include <stdio.h> 
#include <algorithm>
#include <iostream>

// Sorting includes
#include <b40c/util/multi_buffer.cuh>
#include <b40c/radix_sort/enactor.cuh>

// Test utils
#include "b40c_test_util.h"


template <typename T, typename S>
void Assign(T &t, S &s)
{
	t = s;
}

template <typename S>
void Assign(b40c::util::NullType, S &s) {}

template <typename T>
int CastInt(T &t)
{
	return (int) t;
}

int CastInt(b40c::util::NullType)
{
	return 0;
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
//	typedef unsigned long long		KeyType;
//	typedef float					KeyType;
//	typedef char					KeyType;
//	typedef int						KeyType;
	typedef unsigned int 			KeyType;
//	typedef unsigned short 			KeyType;
	typedef b40c::util::NullType 	ValueType;
//	typedef unsigned long long 		ValueType;
//	typedef unsigned int			ValueType;

	static const b40c::radix_sort::ProblemSize PROBLEM_SIZE = b40c::radix_sort::LARGE_PROBLEM;

	const int 		START_BIT			= 0;
	const int 		KEY_BITS 			= sizeof(KeyType) * 8;
	const bool 		KEYS_ONLY			= b40c::util::Equals<ValueType, b40c::util::NullType>::VALUE;
    int 			num_elements 		= 1024 * 1024 * 8;			// 8 million pairs
    unsigned int 	max_ctas 			= 0;						// default: let the enactor decide how many CTAs to launch based upon device properties
    int 			iterations 			= 0;
    int				entropy_reduction 	= 0;
    int 			effective_bits 		= KEY_BITS;

    // Initialize command line
    b40c::CommandLineArgs args(argc, argv);
    b40c::DeviceInit(args);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\nlars_demo [--device=<device index>] [--v] [--n=<elements>] "
    			"[--max-ctas=<max-thread-blocks>] [--i=<iterations>] "
    			"[--zeros | --regular] [--entropy-reduction=<random &'ing rounds>\n");
    	return 0;
    }

    // Parse commandline args
    bool verbose = args.CheckCmdLineFlag("v");
    bool zeros = args.CheckCmdLineFlag("zeros");
    bool regular = args.CheckCmdLineFlag("regular");
    bool schmoo = args.CheckCmdLineFlag("schmoo");
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("i", iterations);
    args.GetCmdLineArgument("max-ctas", max_ctas);
    args.GetCmdLineArgument("entropy-reduction", entropy_reduction);
    args.GetCmdLineArgument("bits", effective_bits);

    // Print header
    if (zeros) printf("Zeros\n");
    else if (regular) printf("%d-bit mod-%llu\n", KEY_BITS, 1ull << effective_bits);
    else printf("%d-bit random\n", KEY_BITS);
    fflush(stdout);

	// Allocate and initialize host problem data and host reference solution.
	// Only use RADIX_BITS effective bits (remaining high order bits
	// are left zero): we only want to perform one sorting pass

    KeyType 	*h_keys 				= new KeyType[num_elements];
	KeyType 	*h_reference_keys 		= new KeyType[num_elements];
    ValueType 	*h_values = NULL;
	if (!KEYS_ONLY) {
		h_values = new ValueType[num_elements];
	}

	if (verbose) printf("Original: ");
	for (int i = 0; i < num_elements; ++i) {

		if (regular) {
			h_keys[i] = i & ((1ull << effective_bits) - 1);
		} else if (zeros) {
			h_keys[i] = 0;
		} else {
			b40c::util::RandomBits(h_keys[i], entropy_reduction, KEY_BITS);
		}
		h_keys[i] *= (1 << START_BIT);

		h_reference_keys[i] = h_keys[i];

		if (!KEYS_ONLY) {
			Assign(h_values[i], i);
		}

		if (verbose) {
			b40c::PrintValue(h_keys[i]);
			printf(", ");
			if ((i & 255) == 255) printf("\n\n");
		}
	}
	if (verbose) printf("\n");

    // Compute reference solution
	std::sort(h_reference_keys, h_reference_keys + num_elements);

	// Allocate device data.
	cudaError_t error;
	b40c::util::DoubleBuffer<KeyType, ValueType> double_buffer;

	error = cudaMalloc((void**) &double_buffer.d_keys[0], sizeof(KeyType) * num_elements);
	if (b40c::util::B40CPerror(error)) exit(1);
	error = cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(KeyType) * num_elements);
	if (b40c::util::B40CPerror(error)) exit(1);
	if (!KEYS_ONLY) {
		error = cudaMalloc((void**) &double_buffer.d_values[0], sizeof(ValueType) * num_elements);
		if (b40c::util::B40CPerror(error)) exit(1);
		error = cudaMalloc((void**) &double_buffer.d_values[1], sizeof(ValueType) * num_elements);
		if (b40c::util::B40CPerror(error)) exit(1);
	}

	// Create a scan enactor
	b40c::radix_sort::Enactor enactor;

	//
	// Perform one sorting pass (starting at bit zero and covering RADIX_BITS bits)
	//

	cudaMemcpy(
		double_buffer.d_keys[double_buffer.selector],
		h_keys,
		sizeof(KeyType) * num_elements,
		cudaMemcpyHostToDevice);
	if (!KEYS_ONLY) {
		cudaMemcpy(
			double_buffer.d_values[double_buffer.selector],
			h_values,
			sizeof(ValueType) * num_elements,
			cudaMemcpyHostToDevice);
	}

	// Sort
	error = enactor.Sort<PROBLEM_SIZE, KEY_BITS, START_BIT>(
		double_buffer, num_elements, 0, max_ctas, true);

	if (error) exit(1);

	printf("\nRestricted-range %s sort (selector %d): ",
		(KEYS_ONLY) ? "keys-only" : "key-value",
		double_buffer.selector);
	fflush(stdout);

	b40c::CompareDeviceResults(
		h_reference_keys,
		double_buffer.d_keys[double_buffer.selector],
		num_elements,
		true,
		verbose); printf("\n");

	if (!KEYS_ONLY)
	{
		cudaMemcpy(
			h_values,
			double_buffer.d_values[double_buffer.selector],
			sizeof(ValueType) * num_elements,
			cudaMemcpyDeviceToHost);

		printf("\n\nValues: ");
		if (verbose) {
			for (int i = 0; i < num_elements; ++i)
			{
				b40c::PrintValue(h_values[i]);
				printf(", ");
			}
			printf("\n\n");
		}

		bool correct = true;
		for (int i = 0; i < num_elements; ++i) {
			if (h_keys[CastInt(h_values[i])] != h_reference_keys[i])
			{
				std::cout << "Incorrect: [" << i << "]: " << h_keys[CastInt(h_values[i])] << " != " << h_reference_keys[i] << std::endl << std::endl;
				correct = false;
				break;
			}
		}
		if (correct) {
			printf("Correct\n\n");
		}
	}

	cudaThreadSynchronize();

	if (schmoo) {
		printf("iteration, elements, elapsed (ms), throughput (MKeys/s)\n");
	}

	b40c::GpuTimer gpu_timer;
	double max_exponent 		= log2(double(num_elements)) - 5.0;
	unsigned int max_int 		= (unsigned int) -1;
	float elapsed 				= 0;

	for (int i = 0; i < iterations; i++) {

		// Reset problem
		double_buffer.selector = 0;
		cudaMemcpy(
			double_buffer.d_keys[double_buffer.selector],
			h_keys,
			sizeof(KeyType) * num_elements,
			cudaMemcpyHostToDevice);
		if (!KEYS_ONLY) {
			cudaMemcpy(
				double_buffer.d_values[double_buffer.selector],
				h_values,
				sizeof(ValueType) * num_elements,
				cudaMemcpyHostToDevice);
		}

		if (schmoo) {

			// Sample a problem size
			unsigned int sample;
			b40c::util::RandomBits(sample);
			double scale = double(sample) / max_int;
			int elements = (i < iterations / 2) ?
				pow(2.0, (max_exponent * scale) + 5.0) :		// log bias
				elements = scale * num_elements;						// uniform bias

			gpu_timer.Start();

			// Sort
			enactor.Sort<PROBLEM_SIZE, KEY_BITS, START_BIT>(
				double_buffer,
				elements,
				0,
				max_ctas);

			gpu_timer.Stop();

			float millis = gpu_timer.ElapsedMillis();
			printf("%d, %d, %.3f, %.2f\n",
				i,
				elements,
				millis,
				float(elements) / millis / 1000.f);
			fflush(stdout);

		} else {

			// Regular iteration
			gpu_timer.Start();

			// Sort
			enactor.Sort<PROBLEM_SIZE, KEY_BITS, START_BIT>(
				double_buffer,
				num_elements,
				0,
				max_ctas);

			gpu_timer.Stop();

			elapsed += gpu_timer.ElapsedMillis();
		}
	}

	// Display output
	if ((!schmoo) && (iterations > 0)) {
		float avg_elapsed = elapsed / float(iterations);
		printf("Elapsed millis: %f, avg elapsed: %f, throughput: %.2f Mkeys/s\n",
			elapsed,
			avg_elapsed,
			float(num_elements) / avg_elapsed / 1000.f);
	}

	// Cleanup device storage
	if (double_buffer.d_keys[0]) cudaFree(double_buffer.d_keys[0]);
	if (double_buffer.d_keys[1]) cudaFree(double_buffer.d_keys[1]);
	if (double_buffer.d_values[0]) cudaFree(double_buffer.d_values[0]);
	if (double_buffer.d_values[1]) cudaFree(double_buffer.d_values[1]);

	// Cleanup other
	delete h_keys;
	delete h_reference_keys;
	delete h_values;

	return 0;
}

