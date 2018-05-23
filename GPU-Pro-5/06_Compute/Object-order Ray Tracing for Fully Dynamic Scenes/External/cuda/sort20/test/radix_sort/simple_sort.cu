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

// Sorting includes
#include <b40c/radix_sort/enactor.cuh>
#include <b40c/util/multi_buffer.cuh>

// Test utils
#include "b40c_test_util.h"


struct Foo
{
	int a[5];
};


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	typedef float KeyType;
	typedef double ValueType;

    unsigned int num_elements = 15;

    b40c::CommandLineArgs args(argc, argv);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\nsimple_sort [--device=<device index>] [--v] [--n=<elements>] [--keys-only]\n");
    	return 0;
    }

    // Parse commandline arguments
    b40c::DeviceInit(args);
    bool verbose = args.CheckCmdLineFlag("v");
    bool keys_only = args.CheckCmdLineFlag("keys-only");
    args.GetCmdLineArgument("n", num_elements);

	// Allocate host problem data
    KeyType *h_keys = new KeyType[num_elements];
    ValueType *h_values = new ValueType[num_elements];
	KeyType *h_reference_keys = new KeyType[num_elements];

	// Initialize host problem data
	if (verbose) printf("Initial problem:\n\n");
	for (int i = 0; i < num_elements; ++i)
	{
		b40c::util::RandomBits(h_keys[i]);
		h_values[i] = i;
		h_reference_keys[i] = h_keys[i];

		if (verbose) {
			b40c::PrintValue(h_keys[i]); printf("("); b40c::PrintValue(h_values[i]); printf("), ");
		}
	}
	if (verbose) printf("\n\n");

    // Compute reference solution
	std::sort(h_reference_keys, h_reference_keys + num_elements);

	// Allocate device data. (We will let the sorting enactor create
	// the "pong" storage if/when necessary.)
	KeyType *d_keys;
	ValueType *d_values;
	cudaMalloc((void**) &d_keys, sizeof(KeyType) * num_elements);
	cudaMalloc((void**) &d_values, sizeof(ValueType) * num_elements);

	// Copy host data to device data
	cudaMemcpy(d_keys, h_keys, sizeof(KeyType) * num_elements, cudaMemcpyHostToDevice);
	cudaMemcpy(d_values, h_values, sizeof(ValueType) * num_elements, cudaMemcpyHostToDevice);

	// Create a reusable sorting enactor
	b40c::radix_sort::Enactor enactor;

	if (keys_only) {

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<KeyType> double_buffer;

		// The current key buffer (double_buffer.d_keys[double_buffer.selector]) backs the keys.
		double_buffer.d_keys[double_buffer.selector] = d_keys;

		// Allocate pong buffer
		cudaMalloc((void**) &double_buffer.d_keys[double_buffer.selector ^ 1], sizeof(KeyType) * num_elements);

		// Sort
		enactor.Sort(double_buffer, num_elements);

		// Check keys answer
		printf("Simple keys-only sort:\n\n");
		b40c::CompareDeviceResults(
			h_reference_keys,
			double_buffer.d_keys[double_buffer.selector],
			num_elements, verbose, verbose);
		printf("\n");

		// Cleanup "pong" storage
		if (double_buffer.d_keys[double_buffer.selector ^ 1]) {
			cudaFree(double_buffer.d_keys[double_buffer.selector ^ 1]);
		}

	} else {

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<KeyType, ValueType> double_buffer;

		// The current key buffer (double_buffer.d_keys[double_buffer.selector]) backs the keys.
		double_buffer.d_keys[double_buffer.selector] = d_keys;
		double_buffer.d_values[double_buffer.selector] = d_values;

		// Allocate pong buffer
		cudaMalloc((void**) &double_buffer.d_keys[double_buffer.selector ^ 1], sizeof(KeyType) * num_elements);
		cudaMalloc((void**) &double_buffer.d_values[double_buffer.selector ^ 1], sizeof(ValueType) * num_elements);

		// Sort
		enactor.Sort(double_buffer, num_elements);

		// Check keys answer
		printf("Simple key-value sort:\n\n: ");
		b40c::CompareDeviceResults(
			h_reference_keys,
			double_buffer.d_keys[double_buffer.selector],
			num_elements, verbose, verbose);

		// Copy out values
		printf("\n\nValues: ");
		cudaMemcpy(h_values, double_buffer.d_values[double_buffer.selector], sizeof(ValueType) * num_elements, cudaMemcpyDeviceToHost);
		if (verbose) {
			for (int i = 0; i < num_elements; ++i)
			{
				b40c::PrintValue(h_values[i]);
				printf(", ");
			}
			printf("\n\n");
		}

		// Check values answer
		bool correct = true;
		for (int i = 0; i < num_elements; ++i)
		{
			if (h_keys[int(h_values[i])] != h_reference_keys[i])
			{
				printf("Incorrect: [%d]: ", i);
				b40c::PrintValue(h_keys[int(h_values[i])]); printf(" != ");
				b40c::PrintValue(h_reference_keys[i]); printf("\n\n");
				correct = false;
				break;
			}
		}
		if (correct) printf("Correct\n\n");

		// Cleanup "pong" storage
		if (double_buffer.d_keys[double_buffer.selector ^ 1]) {
			cudaFree(double_buffer.d_keys[double_buffer.selector ^ 1]);
		}
		if (double_buffer.d_values[double_buffer.selector ^ 1]) {
			cudaFree(double_buffer.d_values[double_buffer.selector ^ 1]);
		}

	}
	
	delete h_keys;
	delete h_values;
	delete h_reference_keys;
	cudaFree(d_keys);
	cudaFree(d_values);

	return 0;
}

