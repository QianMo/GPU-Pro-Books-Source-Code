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

// per-tile light list
groupshared uint ldsLightIdxCounter;
groupshared uint ldsLightIdx[MAX_NUM_LIGHTS_PER_TILE];

// per-tile spot light list
groupshared uint ldsSpotIdxCounter;
groupshared uint ldsSpotIdx[MAX_NUM_LIGHTS_PER_TILE];

//-----------------------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------------------

// Plane equation from three points, simplified 
// for the case where the first point is the origin.
// N is normalized so that the plane equation can 
// be used to compute signed distance.
float4 CreatePlaneEquation(float3 Q, float3 R)
{
    // N = normalize(cross(Q-P,R-P)), 
    // except we know P is the origin
    float3 N = normalize(cross(Q, R));
    // D = -(N dot P), except we know P is the origin
    return float4(N, 0);
}

// Point-plane distance, simplified for the case where 
// the plane passes through the origin
float GetSignedDistanceFromPlane(float3 p, float4 eqn)
{
    // dot(eqn.xyz, p) + eqn.w, except we know eqn.w is zero
    return dot(eqn.xyz, p);
}

void DoLightCulling(in uint3 globalIdx, in uint localIdxFlattened, in uint3 groupIdx)
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
        ldsLightIdxCounter = 0;
        ldsSpotIdxCounter = 0;
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

    float4 frustumEqn[4];
    { // Construct frustum planes for this tile
        uint pxm = TILE_RES*groupIdx.x;
        uint pym = TILE_RES*groupIdx.y;
        uint pxp = TILE_RES*(groupIdx.x + 1);
        uint pyp = TILE_RES*(groupIdx.y + 1);
        uint width = TILE_RES*g_uNumTilesX;
        uint height = TILE_RES*g_uNumTilesY;

        // Four corners of the tile, clockwise from top-left
        float3 p[4];
        p[0] = ConvertProjToView(float4(pxm / (float)width*2.f - 1.f, (height - pym) / (float)height*2.f - 1.f, 1.f, 1.f));
        p[1] = ConvertProjToView(float4(pxp / (float)width*2.f - 1.f, (height - pym) / (float)height*2.f - 1.f, 1.f, 1.f));
        p[2] = ConvertProjToView(float4(pxp / (float)width*2.f - 1.f, (height - pyp) / (float)height*2.f - 1.f, 1.f, 1.f));
        p[3] = ConvertProjToView(float4(pxm / (float)width*2.f - 1.f, (height - pyp) / (float)height*2.f - 1.f, 1.f, 1.f));

        // Create plane equations for the four sides, with 
        // the positive half-space outside the frustum
        for (uint i = 0; i<4; i++)
            frustumEqn[i] = CreatePlaneEquation(p[i], p[(i + 1) & 3]);
    }

    // loop over the lights and do a sphere vs. frustum intersection test
    for (uint i = localIdxFlattened; i<g_uNumLights; i += NUM_THREADS)
    {
        float4 p = g_PointLightBufferCenterAndRadius[i];
        float  r = p.w;
        float3 c = mul(float4(p.xyz, 1), g_mView).xyz;

        // Test if sphere is intersecting or inside frustum
        if ((GetSignedDistanceFromPlane(c, frustumEqn[0]) < r) &&
            (GetSignedDistanceFromPlane(c, frustumEqn[1]) < r) &&
            (GetSignedDistanceFromPlane(c, frustumEqn[2]) < r) &&
            (GetSignedDistanceFromPlane(c, frustumEqn[3]) < r) &&
            (-c.z + minZ < r) && (c.z - maxZ < r))
        {
            // do a thread-safe increment of the list counter 
            // and put the index of this light into the list
            uint dstIdx = 0;
            InterlockedAdd( ldsLightIdxCounter, 1, dstIdx );
            ldsLightIdx[dstIdx] = i;
        }
    }

    // loop over the spot lights and do a sphere vs. frustum intersection test
    for (uint j = localIdxFlattened; j<g_uNumSpotLights; j += NUM_THREADS)
    {
        float4 p = g_SpotLightBufferCenterAndRadius[j];
        float r = p.w;
        float3 c = mul(float4(p.xyz, 1), g_mView).xyz;

        // Test if sphere is intersecting or inside frustum
        if ((GetSignedDistanceFromPlane(c, frustumEqn[0]) < r) &&
            (GetSignedDistanceFromPlane(c, frustumEqn[1]) < r) &&
            (GetSignedDistanceFromPlane(c, frustumEqn[2]) < r) &&
            (GetSignedDistanceFromPlane(c, frustumEqn[3]) < r) &&
            (-c.z + minZ < r) && (c.z - maxZ < r))
        {
            // do a thread-safe increment of the list counter 
            // and put the index of this light into the list
            uint dstIdx = 0;
            InterlockedAdd( ldsSpotIdxCounter, 1, dstIdx );
            ldsSpotIdx[dstIdx] = j;
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

