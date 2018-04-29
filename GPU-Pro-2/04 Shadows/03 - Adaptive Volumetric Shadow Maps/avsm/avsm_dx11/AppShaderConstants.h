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

#ifndef H_APP_SHADER_CONSTANTS
#define H_APP_SHADER_CONSTANTS

#include <fstream>
#include <iostream>
#include <string>

// NOTE: Must match layout of shader constant buffer

__declspec(align(16))
struct UIConstants
{
    unsigned int faceNormals;
    unsigned int avsmSortingMethod;
    unsigned int volumeShadowMethod;
    unsigned int enableVolumeShadowLookup;
    unsigned int pauseParticleAnimaton;
    float        particleSize;
    unsigned int particleOpacity;
    float        dsmError;
    unsigned int hairThickness;
    unsigned int hairShadowThickness;
    unsigned int hairOpacity;
    unsigned int lightingOnly;
};

__declspec(align(16))
struct PerFrameConstants
{
    D3DXMATRIX  mCameraWorldViewProj;
    D3DXMATRIX  mCameraWorldView;
    D3DXMATRIX  mCameraViewProj;
    D3DXMATRIX  mCameraProj;
    D3DXVECTOR4 mCameraPos;
    D3DXMATRIX  mLightWorldViewProj;
    D3DXMATRIX  mAvsmLightWorldViewProj;
    D3DXMATRIX  mCameraViewToLightProj;
    D3DXMATRIX  mCameraViewToLightView;
    D3DXMATRIX  mCameraViewToAvsmLightProj;
    D3DXMATRIX  mCameraViewToAvsmLightView;
    D3DXVECTOR4 mLightDir;

    UIConstants mUI;
};

__declspec(align(16))
struct ParticlePerFrameConstants
{
    float  mScale;                             // Scene scale factor    
    float  mParticleSize;                      // Particles size in (pre)projection space
    float  mParticleOpacity;				   // (Max) Particle contrbution to the CDF
    float  mParticleAlpha;				       // (Max) Particles transparency
    float  mbSoftParticles;				       // Soft Particles Enable-Disable 
    float  mSoftParticlesSaturationDepth;      // Saturation Depth for Soft Particles.      
};

__declspec(align(16))
struct ParticlePerPassConstants
{
    D3DXMATRIX mParticleWorldViewProj;
    D3DXMATRIX mParticleWorldView;
    __declspec(align(16)) D3DXVECTOR3 mEyeRight;
    __declspec(align(16)) D3DXVECTOR3 mEyeUp;   
};

__declspec(align(16))
struct LT_Constants
{
    int   mMaxNodes;    
    float mFirstNodeMapSize;
};

__declspec(align(16))
struct AVSMConstants
{   
    D3DXVECTOR4 mMask0; 
    D3DXVECTOR4 mMask1;
    D3DXVECTOR4 mMask2;
    D3DXVECTOR4 mMask3;
    D3DXVECTOR4 mMask4;
    float       mEmptyNode;
    float       mOpaqueNodeTrans;
    float       mShadowMapSize;
}; 

__declspec(align(16))
struct VolumeShadowConstants
{   
    float mDSMError;    // Deep Shadow Maps error threshold
}; 

// Hair
__declspec(align(16))
struct HairConstants
{
    D3DXMATRIX  mHairProj;        
    D3DXMATRIX  mHairWorldView;
    D3DXMATRIX  mHairWorldViewProj;
};

// Hair
__declspec(align(16))
struct HairLTConstants
{
    unsigned int mMaxNodeCount;
};

#endif // H_APP_SHADER_CONSTANTS
