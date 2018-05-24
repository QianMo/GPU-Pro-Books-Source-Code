//
// Copyright 2014 ADVANCED MICRO DEVICES, INC.  All Rights Reserved.
//
// AMD is granting you permission to use this software and documentation (if
// any) (collectively, the “Materials”) pursuant to the terms and conditions
// of the Software License Agreement included with the Materials.  If you do
// not have a copy of the Software License Agreement, contact your AMD
// representative for a copy.
// You agree that you will not reverse engineer or decompile the Materials,
// in whole or in part, except as allowed by applicable law.
//
// WARRANTY DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND.  AMD DISCLAIMS ALL WARRANTIES, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE, NON-INFRINGEMENT, THAT THE SOFTWARE
// WILL RUN UNINTERRUPTED OR ERROR-FREE OR WARRANTIES ARISING FROM CUSTOM OF
// TRADE OR COURSE OF USAGE.  THE ENTIRE RISK ASSOCIATED WITH THE USE OF THE
// SOFTWARE IS ASSUMED BY YOU.
// Some jurisdictions do not allow the exclusion of implied warranties, so
// the above exclusion may not apply to You. 
// 
// LIMITATION OF LIABILITY AND INDEMNIFICATION:  AMD AND ITS LICENSORS WILL
// NOT, UNDER ANY CIRCUMSTANCES BE LIABLE TO YOU FOR ANY PUNITIVE, DIRECT,
// INCIDENTAL, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES ARISING FROM USE OF
// THE SOFTWARE OR THIS AGREEMENT EVEN IF AMD AND ITS LICENSORS HAVE BEEN
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.  
// In no event shall AMD's total liability to You for all damages, losses,
// and causes of action (whether in contract, tort (including negligence) or
// otherwise) exceed the amount of $100 USD.  You agree to defend, indemnify
// and hold harmless AMD and its licensors, and any of their directors,
// officers, employees, affiliates or agents from and against any and all
// loss, damage, liability and other expenses (including reasonable attorneys'
// fees), resulting from Your use of the Software or violation of the terms and
// conditions of this Agreement.  
//
// U.S. GOVERNMENT RESTRICTED RIGHTS: The Materials are provided with "RESTRICTED
// RIGHTS." Use, duplication, or disclosure by the Government is subject to the
// restrictions as set forth in FAR 52.227-14 and DFAR252.227-7013, et seq., or
// its successor.  Use of the Materials by the Government constitutes
// acknowledgement of AMD's proprietary rights in them.
// 
// EXPORT RESTRICTIONS: The Materials may be subject to export restrictions as
// stated in the Software License Agreement.
//

//--------------------------------------------------------------------------------------
// File: TilingCommonHeader.h
//
// HLSL file for the ComputeBasedTiledCulling sample. Header file for tiled light culling.
//--------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------
// Parameters for the light culling shaders
//-----------------------------------------------------------------------------------------
#define NUM_THREADS (TILE_RES*TILE_RES)

//-----------------------------------------------------------------------------------------
// Textures and Buffers
//-----------------------------------------------------------------------------------------
Buffer<float4>    g_PointLightBufferCenterAndRadius : register( t0 );
Buffer<float4>    g_SpotLightBufferCenterAndRadius  : register( t1 );
Texture2D<float>  g_SceneDepthBuffer                : register( t2 );

//-----------------------------------------------------------------------------------------
// Group Shared Memory (aka local data share, or LDS)
//-----------------------------------------------------------------------------------------
groupshared uint ldsZMin;
groupshared uint ldsZMax;
groupshared uint ldsZMin2;
groupshared uint ldsZMax2;

// per-tile light list
groupshared uint ldsLightIdxCounterA;
groupshared uint ldsLightIdxCounterB;
groupshared uint ldsLightIdx[2 * MAX_NUM_LIGHTS_PER_TILE];

// per-tile spot light list
groupshared uint ldsSpotIdxCounterA;
groupshared uint ldsSpotIdxCounterB;
groupshared uint ldsSpotIdx[2 * MAX_NUM_LIGHTS_PER_TILE];

//-----------------------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------------------

// GPU-friendly version of the squared distance calculation 
// for the Arvo box-sphere intersection test
float ComputeSquaredDistanceToAABB(float3 Pos, float3 AABBCenter, float3 AABBHalfSize)
{
    float3 delta = max(0, abs(AABBCenter - Pos) - AABBHalfSize);
    return dot(delta, delta);
}

// Arvo box-sphere intersection test
bool TestSphereVsAABB(float3 sphereCenter, float sphereRadius, float3 AABBCenter, float3 AABBHalfSize)
{
    float distSq = ComputeSquaredDistanceToAABB(sphereCenter, AABBCenter, AABBHalfSize);
    return distSq <= sphereRadius * sphereRadius;
}

void DoLightCulling(in uint3 globalIdx, in uint localIdxFlattened, in uint3 groupIdx, out float fHalfZ)
{
    float depth = g_SceneDepthBuffer.Load(uint3(globalIdx.x, globalIdx.y, 0)).x;
    float viewPosZ = ConvertProjDepthToView(depth);
    uint z = asuint(viewPosZ);

    // There is no way to initialize shared memory at 
    // compile time, so thread zero does it at run time
    if (localIdxFlattened == 0)
    {
        ldsZMin = 0x7f7fffff;  // FLT_MAX as a uint
        ldsZMax = 0;
        ldsZMin2 = 0x7f7fffff;  // FLT_MAX as a uint
        ldsZMax2 = 0;
        ldsLightIdxCounterA = 0;
        ldsLightIdxCounterB = MAX_NUM_LIGHTS_PER_TILE;
        ldsSpotIdxCounterA = 0;
        ldsSpotIdxCounterB = MAX_NUM_LIGHTS_PER_TILE;
    }
    GroupMemoryBarrierWithGroupSync();

    // Parts of the depth buffer that were never written 
    // (e.g. the sky) will be zero (the companion code uses 
    // inverted 32-bit float depth for better precision).
    if (depth != 0.f)
    {
        // Calculate the min and max depth for this tile, 
        // to form the front and back of the frustum
        InterlockedMin(ldsZMin, z);
        InterlockedMax(ldsZMax, z);
    }
    GroupMemoryBarrierWithGroupSync();

    float minZ = asfloat(ldsZMin);
    float maxZ = asfloat(ldsZMax);
    fHalfZ = 0.5f*(minZ + maxZ);

    if ((depth != 0.f) && (viewPosZ >= fHalfZ))
    {
        // the min on the far side of halfZ
        InterlockedMin(ldsZMin2, z);
    }
    if ((depth != 0.f) && (viewPosZ <= fHalfZ))
    {
        // the max on the near side of halfZ
        InterlockedMax(ldsZMax2, z);
    }
    GroupMemoryBarrierWithGroupSync();

    float minZ2 = asfloat(ldsZMin2);
    float maxZ2 = asfloat(ldsZMax2);

    float3 frustumAABBMid;
    float3 frustumAABBHalfSize;
    float3 frustumAABBMid2;
    float3 frustumAABBHalfSize2;
    { // Construct AABBs for this tile's frustum partition
        uint pxm = TILE_RES*groupIdx.x;
        uint pym = TILE_RES*groupIdx.y;
        uint pxp = TILE_RES*(groupIdx.x + 1);
        uint pyp = TILE_RES*(groupIdx.y + 1);
        uint width = TILE_RES*g_uNumTilesX;
        uint height = TILE_RES*g_uNumTilesY;

        // Two opposite corners of the tile, top-left and bottom-right, at the far plane
        float3 frustumTL = ConvertProjToView(float4(pxm / (float)width*2.f - 1.f, (height - pym) / (float)height*2.f - 1.f, 1.f, 1.f));
        float3 frustumBR = ConvertProjToView(float4(pxp / (float)width*2.f - 1.f, (height - pyp) / (float)height*2.f - 1.f, 1.f, 1.f));

        // First AABB, from minZ to maxZ2 (the near side of halfZ)
        float2 frustumTopLeftAtBack = float2( (maxZ2/frustumTL.z)*frustumTL.x, (maxZ2/frustumTL.z)*frustumTL.y );
        float2 frustumBottomRightAtBack = float2( (maxZ2/frustumBR.z)*frustumBR.x, (maxZ2/frustumBR.z)*frustumBR.y );
        float2 frustumTopLeftAtFront = float2( (minZ/frustumTL.z)*frustumTL.x, (minZ/frustumTL.z)*frustumTL.y );
        float2 frustumBottomRightAtFront = float2( (minZ/frustumBR.z)*frustumBR.x, (minZ/frustumBR.z)*frustumBR.y );

        float2 frustumMinXY = min( frustumTopLeftAtBack, min( frustumBottomRightAtBack, min(frustumTopLeftAtFront, frustumBottomRightAtFront) ) );
        float2 frustumMaxXY = max( frustumTopLeftAtBack, max( frustumBottomRightAtBack, max(frustumTopLeftAtFront, frustumBottomRightAtFront) ) );

        float3 frustumAABBMin = float3(frustumMinXY.x, frustumMinXY.y, minZ);
        float3 frustumAABBMax = float3(frustumMaxXY.x, frustumMaxXY.y, maxZ2);

        frustumAABBMid = (frustumAABBMin + frustumAABBMax) * 0.5f;
        frustumAABBHalfSize = (frustumAABBMax - frustumAABBMin) * 0.5f;

        // Second AABB, from minZ2 to maxZ (the far side of halfZ)
        frustumTopLeftAtBack = float2( (maxZ/frustumTL.z)*frustumTL.x, (maxZ/frustumTL.z)*frustumTL.y );
        frustumBottomRightAtBack = float2( (maxZ/frustumBR.z)*frustumBR.x, (maxZ/frustumBR.z)*frustumBR.y );
        frustumTopLeftAtFront = float2( (minZ2/frustumTL.z)*frustumTL.x, (minZ2/frustumTL.z)*frustumTL.y );
        frustumBottomRightAtFront = float2( (minZ2/frustumBR.z)*frustumBR.x, (minZ2/frustumBR.z)*frustumBR.y );

        frustumMinXY = min( frustumTopLeftAtBack, min( frustumBottomRightAtBack, min(frustumTopLeftAtFront, frustumBottomRightAtFront) ) );
        frustumMaxXY = max( frustumTopLeftAtBack, max( frustumBottomRightAtBack, max(frustumTopLeftAtFront, frustumBottomRightAtFront) ) );

        frustumAABBMin = float3(frustumMinXY.x, frustumMinXY.y, minZ2);
        frustumAABBMax = float3(frustumMaxXY.x, frustumMaxXY.y, maxZ);

        frustumAABBMid2 = (frustumAABBMin + frustumAABBMax) * 0.5f;
        frustumAABBHalfSize2 = (frustumAABBMax - frustumAABBMin) * 0.5f;
    }

    // loop over the lights and do a sphere vs. AABB intersection test
    for (uint i = localIdxFlattened; i<g_uNumLights; i += NUM_THREADS)
    {
        float4 p = g_PointLightBufferCenterAndRadius[i];
        float  r = p.w;
        float3 c = mul(float4(p.xyz, 1), g_mView).xyz;

        // Test if sphere is intersecting or inside AABB
        if (TestSphereVsAABB(c, r, frustumAABBMid, frustumAABBHalfSize))
        {
            // Do a thread-safe increment of the list counter 
            // and put the index of this light into the list
            uint dstIdx = 0;
            InterlockedAdd( ldsLightIdxCounterA, 1, dstIdx );
            ldsLightIdx[dstIdx] = i;
        }
        if (TestSphereVsAABB(c, r, frustumAABBMid2, frustumAABBHalfSize2))
        {
            // Do a thread-safe increment of the list counter 
            // and put the index of this light into the list
            uint dstIdx = 0;
            InterlockedAdd( ldsLightIdxCounterB, 1, dstIdx );
            ldsLightIdx[dstIdx] = i;
        }
    }

    // loop over the spot lights and do a sphere vs. AABB intersection test
    for (uint j = localIdxFlattened; j<g_uNumSpotLights; j += NUM_THREADS)
    {
        float4 p = g_SpotLightBufferCenterAndRadius[j];
        float r = p.w;
        float3 c = mul(float4(p.xyz, 1), g_mView).xyz;

        // Test if sphere is intersecting or inside AABB
        if (TestSphereVsAABB(c, r, frustumAABBMid, frustumAABBHalfSize))
        {
            // Do a thread-safe increment of the list counter 
            // and put the index of this light into the list
            uint dstIdx = 0;
            InterlockedAdd( ldsSpotIdxCounterA, 1, dstIdx );
            ldsSpotIdx[dstIdx] = j;
        }
        if (TestSphereVsAABB(c, r, frustumAABBMid2, frustumAABBHalfSize2))
        {
            // Do a thread-safe increment of the list counter 
            // and put the index of this light into the list
            uint dstIdx = 0;
            InterlockedAdd( ldsSpotIdxCounterB, 1, dstIdx );
            ldsSpotIdx[dstIdx] = j;
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

