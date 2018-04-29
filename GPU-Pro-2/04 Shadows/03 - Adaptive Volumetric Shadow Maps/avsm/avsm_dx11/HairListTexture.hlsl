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

#ifndef H_LIST_TEXTURE_HAIR
#define H_LIST_TEXTURE_HAIR

#define HAIR_LT_NODE_NULL 0xFFFFFFFF

struct HairLTNode
{
    uint   next;
    float  depth;
    float4 color;
};

cbuffer HairLTConstants
{
    uint  mMaxNodeCount;
}

RWTexture2D<uint>              gHairLTFirstNodeOffsetUAV;
RWStructuredBuffer<HairLTNode> gHairLTNodeUAV;
Texture2D<uint>                gHairLTFirstNodeOffsetSRV;
StructuredBuffer<HairLTNode>   gHairLTNodesSRV;

bool HairLTAllocNode(out uint newNodeAddress1D)
{
    // alloc a new node
    newNodeAddress1D = gHairLTNodeUAV.IncrementCounter();
    
    // running out of memory?
    return newNodeAddress1D <= mMaxNodeCount;    
}

// Insert a new node at the head of the list
void HairLTInsertNode(int2 screenAddress, 
                      uint newNodeAddress, 
                      HairLTNode newNode)
{
    uint oldNodeAddress;
    InterlockedExchange(gHairLTFirstNodeOffsetUAV[screenAddress], 
                        newNodeAddress, oldNodeAddress); 
      
    newNode.next = oldNodeAddress;    
    gHairLTNodeUAV[newNodeAddress] = newNode;
}

uint HairLTGetFirstNodeOffset(int2 screenAddress)
{
    return gHairLTFirstNodeOffsetSRV[screenAddress];
}

HairLTNode HairLTGetNode(uint nodeAddress)
{
    return gHairLTNodesSRV[nodeAddress];
}     

#endif // H_LIST_TEXTURE_HAIR
