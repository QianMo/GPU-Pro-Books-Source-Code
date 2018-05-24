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

#ifndef FRAMEBUFFER_FLAT_HLSL
#define FRAMEBUFFER_FLAT_HLSL

// - RGBA 16-bit per component packed into a uint2 per texel
float4 UnpackRGBA16(uint2 e)
{
    return float4(f16tof32(e), f16tof32(e >> 16));
}
uint2 PackRGBA16(float4 c)
{
    return f32tof16(c.rg) | (f32tof16(c.ba) << 16);
}

// Linearize the given 2D address + sample index into our flat framebuffer array
uint GetFramebufferSampleAddress(uint2 coords, uint sampleIndex)
{
    // Major ordering: Row (x), Col (y), MSAA sample
    return (sampleIndex * mFramebufferDimensions.y + coords.y) * mFramebufferDimensions.x + coords.x;
}

#endif // FRAMEBUFFER_FLAT_HLSL
