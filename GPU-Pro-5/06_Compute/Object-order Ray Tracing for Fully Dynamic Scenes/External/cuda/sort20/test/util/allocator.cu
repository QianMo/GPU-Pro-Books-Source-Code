

#include <stdio.h>
#include <iostream>
#include <b40c/util/allocator.cuh>

#include "b40c_test_util.h"

template <typename A, typename B>
void AssertEquals(int line, A a, B b)
{
	if (a != b) {
    	std::cerr << "Line " << line << ": " << a << " != " << b << std::endl;
    	exit(1);
	}
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    b40c::CommandLineArgs args(argc, argv);

    // Parse commandline arguments
    b40c::DeviceInit(args);

    // Get current gpu
    int initial_gpu;
	if (b40c::util::B40CPerror(cudaGetDevice(&initial_gpu))) exit(1);

	int num_gpus;
	if (b40c::util::B40CPerror(cudaGetDeviceCount(&num_gpus))) exit(1);

	// Create allocator with default 6MB-1B allowance
    b40c::util::CachedAllocator allocator;

	printf("Running single-gpu tests...\n"); fflush(stdout);

	//
    // Test1
    //

    // Allocate 5 bytes on the current gpu
    char *d_5B;
    allocator.Allocate((void **) &d_5B, 5);

    // Check that that we have zero bytes allocated on the initial GPU
    AssertEquals(__LINE__, allocator.cached_bytes[initial_gpu], 0);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(__LINE__, allocator.live_blocks.size(), 1);

    //
    // Test2
    //

    // Allocate 4096 bytes on the current gpu
    char *d_4096B;
    allocator.Allocate((void **) &d_4096B, 4096);

    // Check that that we have 2 live blocks on the initial GPU
    AssertEquals(__LINE__, allocator.live_blocks.size(), 2);

    //
    // Test3
    //

    // Deallocate d_5B
    allocator.Deallocate(d_5B);

    // Check that that we have min_bin_bytes free bytes cached on the initial gpu
    AssertEquals(__LINE__, allocator.cached_bytes[initial_gpu], allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(__LINE__, allocator.live_blocks.size(), 1);

    // Check that that we have 1 cached block on the initial GPU
    AssertEquals(__LINE__, allocator.cached_blocks.size(), 1);

    //
    // Test4
    //

    // Deallocate d_4096B
    allocator.Deallocate(d_4096B);

    // Check that that we have the 4096 + min_bin free bytes cached on the initial gpu
    AssertEquals(__LINE__, allocator.cached_bytes[initial_gpu], allocator.min_bin_bytes + 4096);

    // Check that that we have 0 live block on the initial GPU
    AssertEquals(__LINE__, allocator.live_blocks.size(), 0);

    // Check that that we have 2 cached block on the initial GPU
    AssertEquals(__LINE__, allocator.cached_blocks.size(), 2);

    //
    // Test5
    //

    // Allocate 768 bytes on the current gpu
    char *d_768B;
    allocator.Allocate((void **) &d_768B, 768);

    // Check that that we have the min_bin free bytes cached on the initial gpu (4096 was reused)
    AssertEquals(__LINE__, allocator.cached_bytes[initial_gpu], allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(__LINE__, allocator.live_blocks.size(), 1);

    // Check that that we have 1 cached block on the initial GPU
    AssertEquals(__LINE__, allocator.cached_blocks.size(), 1);

    //
    // Test6
    //

    // Allocate max_cached_bytes on the current gpu
    char *d_max_cached;
    allocator.Allocate((void **) &d_max_cached, allocator.max_cached_bytes);

    // Deallocate d_max_cached
    allocator.Deallocate(d_max_cached);

    // Check that that we have the min_bin free bytes cached on the initial gpu (max cached was not returned because we went over)
    AssertEquals(__LINE__, allocator.cached_bytes[initial_gpu], allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(__LINE__, allocator.live_blocks.size(), 1);

    // Check that that we still have 1 cached block on the initial GPU
    AssertEquals(__LINE__, allocator.cached_blocks.size(), 1);

    //
    // Test7
    //

    // Free all cached blocks on all GPUs
    allocator.FreeAllCached();

    // Check that that we have 0 bytes cached on the initial GPU
    AssertEquals(__LINE__, allocator.cached_bytes[initial_gpu], 0);

    // Check that that we have 0 cached blocks across all GPUs
    AssertEquals(__LINE__, allocator.cached_blocks.size(), 0);

    // Check that that still we have 1 live block across all GPUs
    AssertEquals(__LINE__, allocator.live_blocks.size(), 1);

    //
    // Test8
    //

    // Allocate max cached bytes on the current gpu
    allocator.Allocate((void **) &d_max_cached, allocator.max_cached_bytes);

    // Deallocate max cached bytes
    allocator.Deallocate(d_max_cached);

    // Deallocate d_768B
    allocator.Deallocate(d_768B);

    unsigned int power;
    size_t rounded_bytes;
    allocator.NearestPowerOf(power, rounded_bytes, allocator.bin_growth, 768);

    // Check that that we have 4096 free bytes cached on the initial gpu
    AssertEquals(__LINE__, allocator.cached_bytes[initial_gpu], rounded_bytes);

    // Check that that we have 1 cached blocks across all GPUs
    AssertEquals(__LINE__, allocator.cached_blocks.size(), 1);

    // Check that that still we have 0 live block across all GPUs
    AssertEquals(__LINE__, allocator.live_blocks.size(), 0);

    if (num_gpus > 1)
    {
    	printf("Running multi-gpu tests...\n"); fflush(stdout);

        //
        // Test9
        //

    	// Allocate 768 bytes on the next gpu
		int next_gpu = (initial_gpu + 1) % num_gpus;
		char *d_768B_2;
		allocator.Allocate((void **) &d_768B_2, 768, next_gpu);

		// Deallocate d_768B on the next gpu
		allocator.Deallocate(d_768B_2, next_gpu);

		// Check that that we have 4096 free bytes cached on the initial gpu
		AssertEquals(__LINE__, allocator.cached_bytes[initial_gpu], rounded_bytes);

		// Check that that we have 4096 free bytes cached on the second gpu
		AssertEquals(__LINE__, allocator.cached_bytes[next_gpu], rounded_bytes);

	    // Check that that we have 2 cached blocks across all GPUs
	    AssertEquals(__LINE__, allocator.cached_blocks.size(), 2);

	    // Check that that still we have 0 live block across all GPUs
	    AssertEquals(__LINE__, allocator.live_blocks.size(), 0);
    }

    printf("Success\n");
    return 0;
}

