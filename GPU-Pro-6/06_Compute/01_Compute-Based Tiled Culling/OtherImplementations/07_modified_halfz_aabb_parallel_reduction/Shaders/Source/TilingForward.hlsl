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
// File: TilingForward.hlsl
//
// HLSL file for the ComputeBasedTiledCulling sample. Tiled light culling.
//--------------------------------------------------------------------------------------


#include "CommonHeader.h"
#include "TilingCommonHeader.h"

//-----------------------------------------------------------------------------------------
// Textures and Buffers
//-----------------------------------------------------------------------------------------
RWBuffer<uint> g_PerTileLightIndexBufferOut : register( u0 );
RWBuffer<uint> g_PerTileSpotIndexBufferOut  : register( u1 );

//-----------------------------------------------------------------------------------------
// Light culling shader
//-----------------------------------------------------------------------------------------
[numthreads(TILE_RES, TILE_RES, 1)]
void CullLightsCS( uint3 globalIdx : SV_DispatchThreadID, uint3 localIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID )
{
    uint localIdxFlattened = localIdx.x + localIdx.y*TILE_RES;

    // after calling DoLightCulling, the per-tile list of light indices that intersect this tile 
    // will be in ldsLightIdx, and the number of lights that intersect this tile 
    // will be in ldsLightIdxCounterA and ldsLightIdxCounterB
    float fHalfZ;
    DoLightCulling( globalIdx, localIdxFlattened, groupIdx, fHalfZ );

    {   // write back (point lights)
        uint tileIdxFlattened = groupIdx.x + groupIdx.y*g_uNumTilesX;
        uint startOffset = g_uMaxNumElementsPerTile*tileIdxFlattened;

        for( uint i=localIdxFlattened; i<ldsLightIdxCounterA; i+=NUM_THREADS )
        {
            // per-tile list of light indices
            g_PerTileLightIndexBufferOut[startOffset+i+4] = ldsLightIdx[i];
        }

        for( uint j=localIdxFlattened; j<ldsLightIdxCounterB-MAX_NUM_LIGHTS_PER_TILE; j+=NUM_THREADS )
        {
            // per-tile list of light indices
            g_PerTileLightIndexBufferOut[startOffset+j+4+g_uMaxNumLightsPerTile] = ldsLightIdx[j+MAX_NUM_LIGHTS_PER_TILE];
        }

        if( localIdxFlattened == 0 )
        {
            // store fHalfZ for this tile as two 16-bit unsigned values
            uint uHalfZBits = asuint(fHalfZ);
            uint uHalfZBitsHigh = uHalfZBits >> 16;
            uint uHalfZBitsLow = uHalfZBits & 0x0000FFFF;
            g_PerTileLightIndexBufferOut[startOffset+0] = uHalfZBitsHigh;
            g_PerTileLightIndexBufferOut[startOffset+1] = uHalfZBitsLow;

            // store the light count for list A
            g_PerTileLightIndexBufferOut[startOffset+2] = ldsLightIdxCounterA;

            // store the light count for list B
            g_PerTileLightIndexBufferOut[startOffset+3] = ldsLightIdxCounterB-MAX_NUM_LIGHTS_PER_TILE;
        }
    }

    {   // write back (spot lights)
        uint tileIdxFlattened = groupIdx.x + groupIdx.y*g_uNumTilesX;
        uint startOffset = g_uMaxNumElementsPerTile*tileIdxFlattened;

        for( uint i=localIdxFlattened; i<ldsSpotIdxCounterA; i+=NUM_THREADS )
        {
            // per-tile list of light indices
            g_PerTileSpotIndexBufferOut[startOffset+i+4] = ldsSpotIdx[i];
        }

        for( uint j=localIdxFlattened; j<ldsSpotIdxCounterB-MAX_NUM_LIGHTS_PER_TILE; j+=NUM_THREADS )
        {
            // per-tile list of light indices
            g_PerTileSpotIndexBufferOut[startOffset+j+4+g_uMaxNumLightsPerTile] = ldsSpotIdx[j+MAX_NUM_LIGHTS_PER_TILE];
        }

        if( localIdxFlattened == 0 )
        {
            // store fHalfZ for this tile as two 16-bit unsigned values
            uint uHalfZBits = asuint(fHalfZ);
            uint uHalfZBitsHigh = uHalfZBits >> 16;
            uint uHalfZBitsLow = uHalfZBits & 0x0000FFFF;
            g_PerTileSpotIndexBufferOut[startOffset+0] = uHalfZBitsHigh;
            g_PerTileSpotIndexBufferOut[startOffset+1] = uHalfZBitsLow;

            // store the light count for list A
            g_PerTileSpotIndexBufferOut[startOffset+2] = ldsSpotIdxCounterA;

            // store the light count for list B
            g_PerTileSpotIndexBufferOut[startOffset+3] = ldsSpotIdxCounterB-MAX_NUM_LIGHTS_PER_TILE;
        }
    }
}


