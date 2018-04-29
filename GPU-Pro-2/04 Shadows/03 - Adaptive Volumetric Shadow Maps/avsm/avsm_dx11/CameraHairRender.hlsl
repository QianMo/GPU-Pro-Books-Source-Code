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

#include "Common.hlsl"
#include "GBuffer.hlsl"
#include "HairListTexture.hlsl"

#define MAX_NODES 50

typedef FullScreenTriangleVSOut PixelInput;

float4
CameraHairRenderPS(PixelInput input,
                   Texture2D depthBufferSRV) : SV_Target
{
    int2 screenAddress = int2(input.positionViewport.xy);  

    float currentDepth = depthBufferSRV.Load(int3(screenAddress, 0)).x;
    
    // Get offset to the first node
    uint nodeOffset = HairLTGetFirstNodeOffset(screenAddress);   

    // Fetch and sort nodes
    uint nodeCount = 0;
    HairLTNode nodes[MAX_NODES];        
    [loop] while ((nodeOffset != HAIR_LT_NODE_NULL) && (nodeCount < MAX_NODES))
    {
        // Get node..
        HairLTNode node = HairLTGetNode(nodeOffset);

        if (node.depth < currentDepth) {
            // Insertion Sort
            int i = (int) nodeCount;
            while (i > 0) {
                if (nodes[i-1].depth < node.depth) {
                    nodes[i] = nodes[i-1];
                    i--;                            
                } else break;

            }
            nodes[i] = node;

            // Increase node count
            nodeCount++;     
        }

        // Move to next node
        nodeOffset = node.next;                    
    }

    if (nodeCount == 0)
        discard;

    float4 blendColor = float4(0, 0, 0, 0);
    for (uint i = 0; i < nodeCount; ++i) {
        blendColor = nodes[i].color + (1.0f - nodes[i].color.a) * blendColor;
    }

    if (blendColor.a != 0)
	blendColor.rgb /= blendColor.a;

    return blendColor;
}
