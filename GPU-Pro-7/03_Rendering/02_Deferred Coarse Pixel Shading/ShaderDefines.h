/////////////////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or imlied.
// See the License for the specific language governing permissions and
// limitations under the License.
/////////////////////////////////////////////////////////////////////////////////////////////

#ifndef SHADER_DEFINES_H
#define SHADER_DEFINES_H

#define MAX_LIGHTS_POWER 10
#define MAX_LOOP_COUNT   100
#define MAX_LIGHTS (1<<MAX_LIGHTS_POWER)

// This determines the tile size for light binning and associated tradeoffs
#define COMPUTE_SHADER_TILE_GROUP_DIM 32
#define COMPUTE_SHADER_TILE_GROUP_SIZE (COMPUTE_SHADER_TILE_GROUP_DIM*COMPUTE_SHADER_TILE_GROUP_DIM)

// If enabled, defers scheduling of per-pixel-shaded pixels until after top-left pixel 
// has been shaded across the whole tile. This allows better SIMD packing and scheduling.
#define DEFER_PER_PIXEL 1

#endif
