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
#include "CPUT.h"
#include "CPUTLight.h"

// Read light properties from .set file
//-----------------------------------------------------------------------------
CPUTResult CPUTLight::LoadLight(CPUTConfigBlock *pBlock, int *pParentID)
{
    ASSERT( (NULL!=pBlock), _L("Invalid NULL parameter.") );

    CPUTResult result = CPUT_SUCCESS;

    // set the null/group node name
    mName = pBlock->GetValueByName(_L("name"))->ValueAsString();

    // get the parent ID
    *pParentID = pBlock->GetValueByName(_L("parent"))->ValueAsInt();

    LoadParentMatrixFromParameterBlock( pBlock );

    cString lightType = pBlock->GetValueByName(_L("lighttype"))->ValueAsString();
    if(lightType.compare(_L("spot")) == 0)
    {
        mLightParams.nLightType = CPUT_LIGHT_SPOT;
    }
    else if(lightType.compare(_L("directional")) == 0)
    {
        mLightParams.nLightType = CPUT_LIGHT_DIRECTIONAL;
    }
    else if(lightType.compare(_L("point")) == 0)
    {
        mLightParams.nLightType = CPUT_LIGHT_POINT;
    }
    else
    {
        // ASSERT(0,_L(""));
        // TODO: why doesn't assert work here?
    }

    pBlock->GetValueByName(_L("Color"))->ValueAsFloatArray(mLightParams.pColor, 3);
    mLightParams.fIntensity    = pBlock->GetValueByName(_L("Intensity"))->ValueAsFloat();
    mLightParams.fHotSpot      = pBlock->GetValueByName(_L("HotSpot"))->ValueAsFloat();
    mLightParams.fConeAngle    = pBlock->GetValueByName(_L("ConeAngle"))->ValueAsFloat();
    mLightParams.fDecayStart   = pBlock->GetValueByName(_L("DecayStart"))->ValueAsFloat();
    mLightParams.bEnableFarAttenuation = pBlock->GetValueByName(_L("EnableNearAttenuation"))->ValueAsBool();
    mLightParams.bEnableFarAttenuation = pBlock->GetValueByName(_L("EnableFarAttenuation"))->ValueAsBool();
    mLightParams.fNearAttenuationStart = pBlock->GetValueByName(_L("NearAttenuationStart"))->ValueAsFloat();
    mLightParams.fNearAttenuationEnd   = pBlock->GetValueByName(_L("NearAttenuationEnd"))->ValueAsFloat();
    mLightParams.fFarAttenuationStart  = pBlock->GetValueByName(_L("FarAttenuationStart"))->ValueAsFloat();
    mLightParams.fFarAttenuationEnd    = pBlock->GetValueByName(_L("FarAttenuationEnd"))->ValueAsFloat();

    return result;
}
