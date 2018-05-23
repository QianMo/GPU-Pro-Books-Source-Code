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
 * Placeholder for prefixing the b40c namespace for inclusion within other
 * C++ header libraries
 ******************************************************************************/

#pragma once

// For example:
//#define B40C_NS_PREFIX  namespace __thrust_b40c{
//#define B40C_NS_POSTFIX }

#define B40C_NS_PREFIX
#define B40C_NS_POSTFIX



/******************************************************************************
 * Placeholder for CUDART additions that need to be defined for old compiler
 * versions
 ******************************************************************************/

namespace b40c {

#if CUDA_VERSION < 4200

enum cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault   = 0,
    cudaSharedMemBankSizeFourByte  = 1,
    cudaSharedMemBankSizeEightByte = 2
};

inline cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig)
{
	return cudaSuccess;
}

inline cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config)
{
	return cudaSuccess;
}

#endif

} // namespace b40c
