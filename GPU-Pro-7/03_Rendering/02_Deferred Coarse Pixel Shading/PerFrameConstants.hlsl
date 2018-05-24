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

#ifndef PER_FRAME_CONSTANTS_HLSL
#define PER_FRAME_CONSTANTS_HLSL

struct UIConstants
{
    uint forcePerPixel;
    uint lightingOnly;
    uint faceNormals;
    uint visualizeLightCount;
    uint visualizePerSampleShading;
    uint lightCullTechnique;
};

cbuffer PerFrameConstants : register(b0)
{
    float4x4 mCameraWorldViewProj;
    float4x4 mCameraWorldView;
    float4x4 mCameraViewProj;
    float4x4 mCameraProj;
    float4 mCameraNearFar;
    uint4 mFramebufferDimensions;
    
    UIConstants mUI;
};

#endif // PER_FRAME_CONSTANTS_HLSL
