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

#ifndef H_VOLUME_SHADOW
#define H_VOLUME_SHADOW

#include "ListTexture.hlsl"

#define ENABLE_DSM
#define MAX_THREADS_X (2)
#define MAX_THREADS_Y (4)
#define MAX_VISIBILITY_NODE_COUNT    (340)

cbuffer VolumeShadowConstants
{
    float mDSMError;    // Deep Shadow Maps error threshold
}

groupshared uint localAllocator = 0;
groupshared uint localVisNodesOffset[MAX_THREADS_X * MAX_THREADS_Y];
groupshared ListTexPackedVisibilityNode localVisNodes[MAX_VISIBILITY_NODE_COUNT * MAX_THREADS_X * MAX_THREADS_Y];

uint FindLocalVisibilityNode(in uint nodeOffset, in float receiverDepth, out float trans)
{
    // empty list
    if (nodeOffset == NODE_LIST_NULL) {
        trans = 1.0f;
        return NODE_LIST_NULL;
    }
    
    uint prevNodeOffset, nodeCount = 0;
    ListTexVisibilityNode node;
    while (nodeOffset != NODE_LIST_NULL)
    {
        nodeCount++;
        node = UnpackVisibilityNode(localVisNodes[nodeOffset]);        
        
        if (receiverDepth < node.depth)
            break;
        
        prevNodeOffset = nodeOffset;
        nodeOffset =  node.next;   
    }
    
    if (nodeCount == 0) {
        trans = 1.0f;
        return NODE_LIST_NULL;
    }
    
    if (nodeOffset == NODE_LIST_NULL) {
        trans = node.trans;
        return  prevNodeOffset;
    } else {
        if (nodeCount == 1) {
            trans = 1.0f;
            return NODE_LIST_NULL;            
        } else {
            ListTexVisibilityNode prevNode = UnpackVisibilityNode(localVisNodes[prevNodeOffset]);
            trans = LT_Interp(prevNode.depth, node.depth, prevNode.trans, node.trans, receiverDepth);    
            return  prevNodeOffset;
        }
    }
}

void InsertLocalVisibilityNode(in uint threadID, in uint prevNodeAddress, in uint newNodeAddress, in ListTexVisibilityNode newNode)
{
    uint oldValue;
    
    // First node? Insert at the head of the list
    if (prevNodeAddress == NODE_LIST_NULL) 
    {    
        // update pointer to the head of the list and make it point to the new node
        InterlockedExchange(localVisNodesOffset[threadID], newNodeAddress, oldValue);                 

        // Update new node next and store it
        newNode.next = oldValue;                            
        localVisNodes[newNodeAddress] = PackVisibilityNode(newNode);
    }
    // All other cases
    else
    {        
        // Make the previous node point to the new node
        InterlockedExchange(localVisNodes[prevNodeAddress].next, newNodeAddress, oldValue);                 
        
        // Update new node next and store it
        newNode.next  = oldValue;        
        localVisNodes[newNodeAddress] = PackVisibilityNode(newNode);
    }
}

void InsertSegmentVisibility(in uint threadID, in ListTexSegmentNode node)
{
    ListTexVisibilityNode newNode;

    // Segment insertion steps
    // 1) Allocate two new nodes
    // 2) Find where the two nodes are located within the current visibility list
    // 3) Insert them (compositing their transmittance with the contribute given by the current visibility list)
    // 4) Composite segment transmittance with all the nodes located between the segment extremes
    // 5) Composite segment tail with the rest of the visibility curve

    // Allocate two nodes
    uint newNodeOffset[2];
    InterlockedAdd(localAllocator, 1, newNodeOffset[0]);
    InterlockedAdd(localAllocator, 1, newNodeOffset[1]);

    // find insertion location for the first node
    float firstNodeTrans;
    uint  firstNodeOffset = FindLocalVisibilityNode(localVisNodesOffset[threadID], node.depth[0], firstNodeTrans);    
      
    // build first node and insert it
    newNode.depth = node.depth[0];
    newNode.trans = firstNodeTrans;

    InsertLocalVisibilityNode(threadID, firstNodeOffset, newNodeOffset[0], newNode);
    // Increase visibility node count
    uint firstNodeAddress = newNodeOffset[0];

            
    // find insertion location for the second node
    float secondNodeTrans;
    uint  secondNodeOffset = FindLocalVisibilityNode(localVisNodesOffset[threadID], node.depth[1], secondNodeTrans);    
    
    // build second node and insert it    
    newNode.depth = node.depth[1];
    newNode.trans = node.trans * secondNodeTrans;
    
    InsertLocalVisibilityNode(threadID, secondNodeOffset, newNodeOffset[1], newNode);
    // Increase visibility node count        
    uint secondNodeAddress = newNodeOffset[1];         
    
    uint count = 0;
    
    // Composite segment transmittance with each node that lies within the segment
    uint startAddress = localVisNodes[newNodeOffset[0]].next;
    uint endAddress   = newNodeOffset[1];
    while((startAddress != endAddress) && (startAddress != NODE_LIST_NULL))
    {
        // Load node
        newNode  = UnpackVisibilityNode(localVisNodes[startAddress]);
        
        // Update node transmittance (composite with segment trasnmittance)
        newNode.trans *= LT_Interp(node.depth[0], node.depth[1], 1.0f, node.trans, newNode.depth);
        
        // Store node
        localVisNodes[startAddress] = PackVisibilityNode(newNode);
        PackVisibilityNode(newNode);
        // Move to the next node
        startAddress = newNode.next;
        
        count++;
        if (count > 1024)
            break;
    }
            
    // Composite the segment tail with the rest of the visibility curve
    count = 0;
    startAddress = localVisNodes[endAddress].next;
    while (startAddress != NODE_LIST_NULL)
    {
        // Update node transmittance (composite with segment trasnmittance)
        ListTexVisibilityNode unpackedNode = UnpackVisibilityNode(localVisNodes[startAddress]);
        unpackedNode.trans *= node.trans;
        localVisNodes[startAddress] = PackVisibilityNode(unpackedNode);
        
        // Move to the next node
        startAddress = localVisNodes[startAddress].next;
        

        count++;
        if (count > 1024)
            break;            
    }
}

void ComputeSlope(in ListTexVisibilityNode start, in ListTexVisibilityNode end, in float error, out float slope[2])
{
    float delta = end.depth - start.depth;
    slope[0] = (end.trans - error - start.trans) / delta;
    slope[1] = (end.trans + error - start.trans) / delta;
}

[numthreads(MAX_THREADS_X, MAX_THREADS_Y, 1)]
void ComputeVisibilityCurveCS(uint3 groupId        : SV_GroupID,
                              uint3 groupThreadId  : SV_GroupThreadID,
                              uint  groupIndex     : SV_GroupIndex)
{
    uint i;
    
    // Get thread ID
    uint threadID = groupIndex;
    
    // Initialize list head offset
    localVisNodesOffset[threadID] = NODE_LIST_NULL;
    
    // Get fragment viewport coordinates
    int2 screenAddress = (int2(MAX_THREADS_X, MAX_THREADS_Y) * groupId.xy) + groupThreadId.xy;      

    //////////////////////////////////////////////
    // Phase 1: Create Visibility Curve
    //////////////////////////////////////////////
        
    // Get offset to the first node
    uint nodeOffset = LT_GetFirstSegmentNodeOffset(screenAddress);
    // Early exit
    if (nodeOffset == NODE_LIST_NULL)
        return;

    // Fetch nodes
    ListTexSegmentNode node;   
    // N.B: DO NOT replace this for loop with a while loop, this is work-around for a back in the software stack (HLSL compiler or ATI drivers)
    for (uint pippo = 0; pippo < MAX_VISIBILITY_NODE_COUNT / 2; pippo++)
    {        
        if (nodeOffset != NODE_LIST_NULL)
        {
            // Get a new segment
            node = LT_GetSegmentNode(nodeOffset);
                       
            // Insert segment in the visibility list
            InsertSegmentVisibility(threadID, node);            

            // Increase node count and move to next segment
            nodeOffset = node.next;   
        }
        else break;                                           
    } 
        
#ifdef ENABLE_DSM
    
    //////////////////////////////////////////////
    // Phase 2: Compress and Store Visibility Data
    //////////////////////////////////////////////    
       
    const float error = 0.0000000000001f;
    uint  endNodeAddress;
    float slope[2];
        
    ListTexVisibilityNode startNode, endNode;
    uint srcNodeAddress = localVisNodesOffset[threadID];    
    
    // Get first node
    startNode      = UnpackVisibilityNode(localVisNodes[srcNodeAddress]);
    srcNodeAddress = startNode.next;
    
    //  Insert it at the head of the list
    uint newNodeAddress;
    if (LT_AllocVisibilityNode(newNodeAddress)) {
        LT_InsertFirstVisibilityNode(screenAddress, newNodeAddress, startNode);   
    } else return;
        
    int phaseID = 1;
    uint tailNodeAddress;
    while (srcNodeAddress != NODE_LIST_NULL)
    {
        tailNodeAddress = newNodeAddress;
        if (phaseID == 0)
        {
            startNode  = endNode;
        }
        else if (phaseID == 1)
        {
            // Fetch current end node
            endNode         = UnpackVisibilityNode(localVisNodes[srcNodeAddress]);
            
            // move to the next node  
            srcNodeAddress  = endNode.next;
             
            ComputeSlope(startNode, endNode, mDSMError, slope);
        }
        else
        {
            float newSlope[2];
            ListTexVisibilityNode newEndNode = UnpackVisibilityNode(localVisNodes[srcNodeAddress]);
            uint nextNodeAddress = newEndNode.next;
            ComputeSlope(startNode, newEndNode, mDSMError, newSlope);
            
            if ((nextNodeAddress != NODE_LIST_NULL) &&
                (slope[0] <= newSlope[1]) &&
                (slope[1] >= newSlope[0]))
            {
                slope[0] = max(slope[0], newSlope[0]);
                slope[1] = min(slope[1], newSlope[1]);  
                              
                endNode = newEndNode;
                srcNodeAddress = nextNodeAddress;
            }
            else
            {
                endNode.trans = max(0.0f, startNode.trans + 0.5f * (endNode.depth - startNode.depth) * (slope[0] + slope[1]));
                
                if (LT_AllocVisibilityNode(newNodeAddress)) {
                    LT_InsertLastVisibilityNode(tailNodeAddress, newNodeAddress, endNode);   
                } else return;     
                
                phaseID = -1;               
            }

        }            
        phaseID++;               
    }
    
    if (phaseID) 
    {
        if (phaseID > 2) {
            endNode.trans = max(0.0f, startNode.trans + 0.5f * (endNode.depth - startNode.depth) * (slope[0] + slope[1]));
        } 
        
        endNode.next  = NODE_LIST_NULL;
        
        if (LT_AllocVisibilityNode(newNodeAddress)) {
            LT_InsertLastVisibilityNode(tailNodeAddress, newNodeAddress, endNode);   
        }                 
    }
  
#else
  
    //////////////////////////////////////////////
    // Phase 2: Pack&Store Visibility Curve Data
    //////////////////////////////////////////////
    
    uint newNodeAddress;            
    uint visNodeAddress = localVisNodesOffset[threadID];  
    ListTexVisibilityNode localNode;
    
    // insert head of the list
    if (visNodeAddress != NODE_LIST_NULL) 
    {
        // Allocate a new node in the visibility list        
        if (LT_AllocVisibilityNode(newNodeAddress)) 
        {        
            localNode = UnpackVisibilityNode(localVisNodes[visNodeAddress]);
                       
            ListTexVisibilityNode visNode;
            visNode.next  = NODE_LIST_NULL;
            visNode.depth = localNode.depth;
            visNode.trans = localNode.trans;
            
            //  Insert it at the head of the list
            LT_InsertFirstVisibilityNode(screenAddress, newNodeAddress, visNode);            
        }
        // move to the next node
        visNodeAddress = localNode.next;
    }
    
    // insert remaining nodes
    // N.B. we don't use some unified push_back() code in order to remove checks for head of the list
    // that are only strictly necessary for the first node. Code is sligtly less elegant but certainly faster
    while (visNodeAddress != NODE_LIST_NULL)
    {
        uint tailNodeAddress = newNodeAddress;

        // Allocate a new node in the visibility list        
        if (LT_AllocVisibilityNode(newNodeAddress)) 
        {        
            // Load local visibility node        
            localNode = UnpackVisibilityNode(localVisNodes[visNodeAddress]);
            
            ListTexVisibilityNode visNode;
            visNode.next  = NODE_LIST_NULL;
            visNode.depth = localNode.depth;
            visNode.trans = localNode.trans;
            
            // Insert it at the head of the list
            LT_InsertLastVisibilityNode(tailNodeAddress, newNodeAddress, visNode);            
        }       
       
        // move to the next node
        visNodeAddress = localNode.next;       
    }       
#endif // ENABLE_DSM

}

#endif
