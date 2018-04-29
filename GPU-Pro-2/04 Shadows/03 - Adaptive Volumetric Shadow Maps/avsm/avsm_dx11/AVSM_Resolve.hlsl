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

#ifndef H_AVSM_RESOLVE
#define H_AVSM_RESOLVE

#include "AVSM.hlsl"

#define MAX_NODES 300

void InitAVSMData(inout AVSMData data)
{
    data.depth[0] = mEmptyNode.xxxx;
    data.trans[0] = FIRST_NODE_TRANS_VALUE.xxxx;
#if AVSM_RT_COUNT > 1    
    data.depth[1] = mEmptyNode.xxxx;
    data.trans[1] = FIRST_NODE_TRANS_VALUE.xxxx;  
#endif    
#if AVSM_RT_COUNT > 2    
    data.depth[2] = mEmptyNode.xxxx;
    data.trans[2] = FIRST_NODE_TRANS_VALUE.xxxx;   
#endif    
#if AVSM_RT_COUNT > 3    
    data.depth[3] = mEmptyNode.xxxx;
    data.trans[3] = FIRST_NODE_TRANS_VALUE.xxxx;  
#endif    
}

// These resolve functions read the previously captured (in a per pixel linked list) light blockers, 
// sort them and insert them in our AVSM
// Note that AVSMs can be created submitting blockers in any order, in this particular case we might want to sort the blockers
// only the reduce temporal artifacts introduced by the non deterministic fragments shading order and AVSM lossy compression algorithm

AVSMData_PSOut AVSMUnsortedResolvePS(FullScreenTriangleVSOut Input)
{       
    AVSMData data;

    // Initialize AVSM data    
    InitAVSMData(data);
        
    // Get fragment viewport coordinates
    int2 screenAddress = int2(Input.positionViewport.xy);  
    
    // Get offset to the first node
    uint nodeOffset = LT_GetFirstSegmentNodeOffset(screenAddress);   

    // Fetch nodes
    ListTexSegmentNode node;  
    [loop]while (nodeOffset != NODE_LIST_NULL) {
        // Get node..
        node = LT_GetSegmentNode(nodeOffset);
        
        // Insert this node into our AVSM
        InsertSegmentAVSM(node.depth, node.trans, data);     
        
        // Increase node count and move to next node
        nodeOffset = node.next;                    
    } 
    
    return data;    
}

AVSMData_PSOut AVSMInsertionSortResolvePS(FullScreenTriangleVSOut Input)
{       
    AVSMData data;

    // Initialize AVSM data    
    InitAVSMData(data);
        
    // Get fragment viewport coordinates
    int2 screenAddress = int2(Input.positionViewport.xy);  
    
    // Get offset to the first node
    uint nodeOffset = LT_GetFirstSegmentNodeOffset(screenAddress);   

    // Fetch nodes
    uint nodeCount = 0;
    ListTexSegmentNode nodes[MAX_NODES];        
    [loop]while ((nodeOffset != NODE_LIST_NULL) && (nodeCount < MAX_NODES)) {
        // Get node..
        ListTexSegmentNode node = LT_GetSegmentNode(nodeOffset);
        
        // Insertion Sort
        int i = (int)nodeCount;
        while (i > 0) {
            if (nodes[i-1].sortKey < node.sortKey) {
                nodes[i] = nodes[i-1];
                i--;                            
            } else break;
            
        }
        nodes[i] = node;
        
        // Increase node count and move to next node
        nodeOffset = node.next;                    
        nodeCount++;     
    } 
   
    // Insert nodes into our AVSM
    [loop]for (uint i = 0; i < nodeCount; ++i) {        
        InsertSegmentAVSM(nodes[i].depth, nodes[i].trans, data);     
    }        
 
    return data;    
}

//////////////////////////////////////////////
// Other algorithms
//////////////////////////////////////////////

#endif // H_AVSM_RESOLVE