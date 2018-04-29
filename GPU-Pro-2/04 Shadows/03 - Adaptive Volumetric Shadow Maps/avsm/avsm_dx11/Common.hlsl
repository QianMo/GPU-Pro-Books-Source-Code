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

#ifndef H_COMMON
#define H_COMMON

//////////////////////////////////////////////
// Defines
//////////////////////////////////////////////

#define AVSM_FILTERING_ENABLED
#define LT_BILINEAR_FILTERING

//////////////////////////////////////////////
// Full screen pass
//////////////////////////////////////////////

struct FullScreenTriangleVSOut
{
    float4 positionViewport : SV_Position;
    float4 positionClip     : positionClip;
    float2 texCoord         : texCoord;
};

//////////////////////////////////////////////
// Particle renderer
//////////////////////////////////////////////

struct DynamicParticlePSIn
{
    float4 Position  : SV_POSITION;
    float3 UVS		 : TEXCOORD0;
    float  Opacity	 : TEXCOORD1;
    float3 ViewPos	 : TEXCOORD2;
    float3 ObjPos    : TEXCOORD3;
    float3 ViewCenter: TEXCOORD4;
};

struct ParticlePSOut
{ 
    float4 color      : COLOR;
};

//////////////////////////////////////////////
// Helper Functions
//////////////////////////////////////////////

float linstep(float min, float max, float v)
{
    return saturate((v - min) / (max - min));
}

#endif // H_COMMON