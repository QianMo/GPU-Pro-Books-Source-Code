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
#ifndef __CPUTLight_H__
#define __CPUTLight_H__

#include "CPUT.h"
#include "CPUTRenderNode.h"
#include "CPUTConfigBlock.h"

enum LightType
{
    CPUT_LIGHT_DIRECTIONAL,
    CPUT_LIGHT_POINT,
    CPUT_LIGHT_SPOT,
};

struct CPUTLightParams
{
    LightType   nLightType;
    float       pColor[3];
    float       fIntensity;
    float       fHotSpot;
    float       fConeAngle;
    float       fDecayStart;
    bool        bEnableNearAttenuation;
    bool        bEnableFarAttenuation;
    float       fNearAttenuationStart;
    float       fNearAttenuationEnd;
    float       fFarAttenuationStart;
    float       fFarAttenuationEnd;
};

class CPUTLight:public CPUTRenderNode
{
protected:
    CPUTLightParams mLightParams;
public:
    CPUTLight() {}
    virtual ~CPUTLight() {}

    void             SetLightParameters(CPUTLightParams &lightParams);
    CPUTLightParams *GetLightParameters() {return &mLightParams; }
    CPUTResult       LoadLight(CPUTConfigBlock *pBlock, int *pParentID);
};

#endif //#ifndef __CPUTLight_H__