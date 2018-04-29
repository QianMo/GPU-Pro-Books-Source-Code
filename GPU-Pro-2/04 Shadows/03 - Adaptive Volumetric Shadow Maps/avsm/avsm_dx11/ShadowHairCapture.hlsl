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

#include "GBuffer.hlsl"
#include "HairCommon.hlsl"
#include "ListTexture.hlsl"

// #define SORT_ON_PRIMITIVE_ID

void ShadowHairModel(in HairPixelInput input, 
                     out float3 entry, 
                     out float3 exit, 
                     out float segmentTransmittance)
{
    // Need to make this be a parameter of each hair strand
    const float hairThickness = float(mUI.hairShadowThickness) / 100.0f;

    const float hairOpacity = ComputeHairOpacity(input.distanceFromCenter);

    // Need to calculate ray-cylinder intersection here to find
    // entry/exit points
    entry = input.position;
    exit = entry + float3(0, 0, hairThickness);
    segmentTransmittance = 1 - hairOpacity;
}

[earlydepthstencil]
void
ShadowHairCapturePS(HairPixelInput input
#ifdef SORT_ON_PRIMITIVE_ID
		    , uint primitiveID : SV_PrimitiveID
#endif
		    )
{
    float3 entry, exit;
    float  segmentTransmittance;    
    ShadowHairModel(input, entry, exit, segmentTransmittance);

    // Allocate a new node
    // (If we're running out of memory we simply drop this fragment
    uint newNodeAddress;
    if (LT_AllocSegmentNode(newNodeAddress)) {
        // Fill node
        ListTexSegmentNode node;
        node.depth[0] = entry.z;
        node.depth[1] = exit.z;
        node.trans    = segmentTransmittance;
#ifdef SORT_ON_PRIMITIVE_ID
        node.sortKey  = primitiveID;
#else
        node.sortKey  = input.position.z;
#endif

        // Get fragment viewport coordinates
        int2 screenAddress = int2(input.hposition.xy);

        // Insert node!
        LT_InsertFirstSegmentNode(screenAddress, newNodeAddress, node);
    }
}


