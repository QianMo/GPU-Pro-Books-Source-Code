// Copyright 2010 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.

#ifndef HEADER_AVSM_DEFINE
#define HEADER_AVSM_DEFINE

// Support for 4, 8, 12, 16 nodes.
#ifndef AVSM_NODE_COUNT
#define AVSM_NODE_COUNT (8)         
#endif

// Enable bilinear filtering (disable it to enable point filtering)
#define AVSM_BILINEARF

// Defines the compositing operator according
// the selected visibily curve representation (transmittance or opacity)
// Also defines the starting value for transmittance and opacity
#define FIRST_NODE_TRANS_VALUE 1.0f

#define VOL_SHADOW_NO_SHADOW         0
#define VOL_SHADOW_UNCOMPRESSED      1
#define VOL_SHADOW_DSM               2
#define VOL_SHADOW_AVSM              3
#define VOL_SHADOW_AVSM_BOX_4        6
#define VOL_SHADOW_AVSM_GAUSS_7      7

// Use these #define to enable/disable particular sampling techniques 
// In order to measure accurate performance for each algorithm all other algorithms must be disabled
// It also improves shaders compile time by a lot if you are only working on a single algorithm!
#define ENABLE_DSM_SAMPLING
#define ENABLE_UNCOMPRESSED_SAMPLING
#define ENABLE_AVSM_SAMPLING 
#define ENABLE_AVSM_BOX4X4_SAMPLING
#define ENABLE_AVSM_GAUSS7X7_SAMPLING

// Select ONLY ONE sampling algorithm (see aboce) and uncomment this #define if you want to enable faster filtering
//#define SAMPLING_NO_SWITCHCASE

#define HAIR_SORT_NONE      0
#define HAIR_SORT_PER_LINE  1
#define HAIR_SORT_PER_PIXEL 2

// By enabling this macro, the shader code will record memory stats
// that can be queried by the app and toggled on/off via the UI.
// #define AVSM_ENABLE_MEMORY_STATS

#endif   
