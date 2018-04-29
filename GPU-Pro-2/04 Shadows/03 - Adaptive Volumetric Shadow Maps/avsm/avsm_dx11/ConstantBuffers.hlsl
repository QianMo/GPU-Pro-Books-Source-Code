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

#ifndef H_CONSTANT_BUFFERS
#define H_CONSTANT_BUFFERS

///////////////////////
// Constants
///////////////////////

cbuffer ParticlePerFrameConstants
{
    float  mScale;                             // Scene scale factor    
    float  mParticleSize;                      // Particles size in (pre)projection space
    float  mParticleOpacity;				   // (Max) Particle contrbution to the CDF
    float  mParticleAlpha;				       // (Max) Particles transparency
    float  mbSoftParticles;				       // Soft Particles Enable-Disable 
    float  mSoftParticlesSaturationDepth;      // Saturation Depth for Soft Particles.      
}

cbuffer ParticlePerPassConstants
{
    float4x4  mParticleWorldViewProj;
    float4x4  mParticleWorldView;
    float4    mEyeRight;
    float4    mEyeUp;       
}

#endif // H_CONSTANT_BUFFERS
