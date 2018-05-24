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
// File: ParallelReduction.hlsl
//
// HLSL file for the ComputeBasedTiledCulling sample. Calculate per-tile depth bounds 
// using parallel reduction.
//--------------------------------------------------------------------------------------


#include "CommonHeader.h"

#define FLT_MAX         3.402823466e+38F

#define NUM_THREADS_1D (TILE_RES/2)
#define NUM_THREADS (NUM_THREADS_1D*NUM_THREADS_1D)

//-----------------------------------------------------------------------------------------
// Textures and Buffers
//-----------------------------------------------------------------------------------------
Texture2D<float>    g_SceneDepthBuffer : register( t0 );
RWTexture2D<float4> g_DepthBounds      : register( u0 );

//-----------------------------------------------------------------------------------------
// Group Shared Memory (aka local data share, or LDS)
//-----------------------------------------------------------------------------------------
// min and max depth per tile
groupshared float ldsZMin[NUM_THREADS];
groupshared float ldsZMax[NUM_THREADS];

[numthreads(NUM_THREADS_1D,NUM_THREADS_1D,1)]
void CalculateDepthBoundsCS( uint3 globalIdx : SV_DispatchThreadID, uint3 localIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID )
{
    uint2 sampleIdx = globalIdx.xy*2;

    // Load four depth samples
    float depth00 = g_SceneDepthBuffer.Load(uint3(sampleIdx.x,  sampleIdx.y,  0)).x;
    float depth01 = g_SceneDepthBuffer.Load(uint3(sampleIdx.x,  sampleIdx.y+1,0)).x;
    float depth10 = g_SceneDepthBuffer.Load(uint3(sampleIdx.x+1,sampleIdx.y,  0)).x;
    float depth11 = g_SceneDepthBuffer.Load(uint3(sampleIdx.x+1,sampleIdx.y+1,0)).x;

    float viewPosZ00 = ConvertProjDepthToView(depth00);
    float viewPosZ01 = ConvertProjDepthToView(depth01);
    float viewPosZ10 = ConvertProjDepthToView(depth10);
    float viewPosZ11 = ConvertProjDepthToView(depth11);

    uint threadNum = localIdx.x + localIdx.y*NUM_THREADS_1D;

    // Use parallel reduction to calculate the depth bounds
    {
        // Parts of the depth buffer that were never written 
        // (e.g. the sky) will be zero (the companion code uses 
        // inverted 32-bit float depth for better precision).
        float minZ00 = (depth00 != 0.f) ? viewPosZ00 : FLT_MAX;
        float minZ01 = (depth01 != 0.f) ? viewPosZ01 : FLT_MAX;
        float minZ10 = (depth10 != 0.f) ? viewPosZ10 : FLT_MAX;
        float minZ11 = (depth11 != 0.f) ? viewPosZ11 : FLT_MAX;

        float maxZ00 = (depth00 != 0.f) ? viewPosZ00 : 0.0f;
        float maxZ01 = (depth01 != 0.f) ? viewPosZ01 : 0.0f;
        float maxZ10 = (depth10 != 0.f) ? viewPosZ10 : 0.0f;
        float maxZ11 = (depth11 != 0.f) ? viewPosZ11 : 0.0f;

        // Initialize shared memory
        ldsZMin[threadNum] = min(minZ00,min(minZ01,min(minZ10,minZ11)));
        ldsZMax[threadNum] = max(maxZ00,max(maxZ01,max(maxZ10,maxZ11)));
        GroupMemoryBarrierWithGroupSync();

        // Min and max using parallel reduction, with the loop manually unrolled for 
        // 8x8 thread groups (64 threads per thread group)
        if (threadNum < 32)
        {
            ldsZMin[threadNum] = min(ldsZMin[threadNum],ldsZMin[threadNum+32]);
            ldsZMax[threadNum] = max(ldsZMax[threadNum],ldsZMax[threadNum+32]);
            ldsZMin[threadNum] = min(ldsZMin[threadNum],ldsZMin[threadNum+16]);
            ldsZMax[threadNum] = max(ldsZMax[threadNum],ldsZMax[threadNum+16]);
            ldsZMin[threadNum] = min(ldsZMin[threadNum],ldsZMin[threadNum+8]);
            ldsZMax[threadNum] = max(ldsZMax[threadNum],ldsZMax[threadNum+8]);
            ldsZMin[threadNum] = min(ldsZMin[threadNum],ldsZMin[threadNum+4]);
            ldsZMax[threadNum] = max(ldsZMax[threadNum],ldsZMax[threadNum+4]);
            ldsZMin[threadNum] = min(ldsZMin[threadNum],ldsZMin[threadNum+2]);
            ldsZMax[threadNum] = max(ldsZMax[threadNum],ldsZMax[threadNum+2]);
            ldsZMin[threadNum] = min(ldsZMin[threadNum],ldsZMin[threadNum+1]);
            ldsZMax[threadNum] = max(ldsZMax[threadNum],ldsZMax[threadNum+1]);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    float minZ = ldsZMin[0];
    float maxZ = ldsZMax[0];
    float halfZ = 0.5f*(minZ + maxZ);

    // Calculate a second set of depth bounds, with the min on the far side of halfZ
    // and the max on the near side of halfZ 
    {
        // We want the min on the far side of halfZ and the max on the near side of halfZ
        float minZ00 = ( viewPosZ00 >= halfZ ) ? viewPosZ00 : FLT_MAX;
        float minZ01 = ( viewPosZ01 >= halfZ ) ? viewPosZ01 : FLT_MAX;
        float minZ10 = ( viewPosZ10 >= halfZ ) ? viewPosZ10 : FLT_MAX;
        float minZ11 = ( viewPosZ11 >= halfZ ) ? viewPosZ11 : FLT_MAX;

        float maxZ00 = ( viewPosZ00 <= halfZ ) ? viewPosZ00 : 0.0f;
        float maxZ01 = ( viewPosZ01 <= halfZ ) ? viewPosZ01 : 0.0f;
        float maxZ10 = ( viewPosZ10 <= halfZ ) ? viewPosZ10 : 0.0f;
        float maxZ11 = ( viewPosZ11 <= halfZ ) ? viewPosZ11 : 0.0f;

        // Initialize shared memory
        ldsZMin[threadNum] = min(minZ00,min(minZ01,min(minZ10,minZ11)));
        ldsZMax[threadNum] = max(maxZ00,max(maxZ01,max(maxZ10,maxZ11)));
        GroupMemoryBarrierWithGroupSync();

        // Min and max using parallel reduction, with the loop manually unrolled for 
        // 8x8 thread groups (64 threads per thread group)
        if (threadNum < 32)
        {
            ldsZMin[threadNum] = min(ldsZMin[threadNum],ldsZMin[threadNum+32]);
            ldsZMax[threadNum] = max(ldsZMax[threadNum],ldsZMax[threadNum+32]);
            ldsZMin[threadNum] = min(ldsZMin[threadNum],ldsZMin[threadNum+16]);
            ldsZMax[threadNum] = max(ldsZMax[threadNum],ldsZMax[threadNum+16]);
            ldsZMin[threadNum] = min(ldsZMin[threadNum],ldsZMin[threadNum+8]);
            ldsZMax[threadNum] = max(ldsZMax[threadNum],ldsZMax[threadNum+8]);
            ldsZMin[threadNum] = min(ldsZMin[threadNum],ldsZMin[threadNum+4]);
            ldsZMax[threadNum] = max(ldsZMax[threadNum],ldsZMax[threadNum+4]);
            ldsZMin[threadNum] = min(ldsZMin[threadNum],ldsZMin[threadNum+2]);
            ldsZMax[threadNum] = max(ldsZMax[threadNum],ldsZMax[threadNum+2]);
            ldsZMin[threadNum] = min(ldsZMin[threadNum],ldsZMin[threadNum+1]);
            ldsZMax[threadNum] = max(ldsZMax[threadNum],ldsZMax[threadNum+1]);
        }
    }

    // Have the first thread write out to the depth bounds texture
    if(threadNum == 0)
    {
        float minZ2 = ldsZMin[0];
        float maxZ2 = ldsZMax[0];
        g_DepthBounds[groupIdx.xy] = float4(minZ,maxZ2,minZ2,maxZ);
    }
}
