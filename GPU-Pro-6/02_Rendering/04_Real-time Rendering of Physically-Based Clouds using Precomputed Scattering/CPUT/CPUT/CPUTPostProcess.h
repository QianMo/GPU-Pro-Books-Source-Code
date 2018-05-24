//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
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
//--------------------------------------------------------------------------------------
#ifndef _CPUTPOSTPROCESS_H
#define _CPUTPOSTPROCESS_H

class CPUTRenderTargetColor;
class CPUTMaterial;
class CPUTRenderParameters;
class CPUTSprite;

class CPUTPostProcess
{
protected:
    CPUTRenderTargetColor *mpRTSourceRenderTarget;
    CPUTRenderTargetColor *mpRTDownSample4x4;
    CPUTRenderTargetColor *mpRTDownSample4x4PingPong;
    CPUTRenderTargetColor *mpRT64x64;
    CPUTRenderTargetColor *mpRT4x4;
    CPUTRenderTargetColor *mpRT1x1;

    CPUTMaterial *mpMaterialSpriteNoAlpha;
    CPUTMaterial *mpMaterialDownSampleBackBuffer4x4;
    CPUTMaterial *mpMaterialDownSample4x4;
    CPUTMaterial *mpMaterialDownSample4x4Alpha;
    CPUTMaterial *mpMaterialDownSampleLogLum;
    CPUTMaterial *mpMaterialBlurHorizontal;
    CPUTMaterial *mpMaterialBlurVertical;
    CPUTMaterial *mpMaterialComposite;

    CPUTSprite   *mpFullScreenSprite;

public:
    CPUTPostProcess() :
        mpRTSourceRenderTarget(NULL),
        mpRTDownSample4x4(NULL),
        mpRTDownSample4x4PingPong(NULL),
        mpRT64x64(NULL),
        mpRT4x4(NULL),
        mpRT1x1(NULL),
        mpMaterialSpriteNoAlpha(NULL),
        mpMaterialDownSampleBackBuffer4x4(NULL),
        mpMaterialDownSample4x4(NULL),
        mpMaterialDownSample4x4Alpha(NULL),
        mpMaterialDownSampleLogLum(NULL),
        mpMaterialBlurHorizontal(NULL),
        mpMaterialBlurVertical(NULL),
        mpMaterialComposite(NULL),
        mpFullScreenSprite(NULL)
    {}
    ~CPUTPostProcess();

    void CreatePostProcess( CPUTRenderTargetColor *pSourceRenderTarget );
    void PerformPostProcess(CPUTRenderParameters &renderParams);
};

#endif // _CPUTPOSTPROCESS_H
