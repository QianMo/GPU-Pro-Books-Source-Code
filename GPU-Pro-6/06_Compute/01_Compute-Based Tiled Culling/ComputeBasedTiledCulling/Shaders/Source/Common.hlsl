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
// File: Common.hlsl
//
// HLSL file for the ComputeBasedTiledCulling sample. Common shaders.
//--------------------------------------------------------------------------------------


#include "CommonHeader.h"


//-----------------------------------------------------------------------------------------
// Textures and Buffers
//-----------------------------------------------------------------------------------------

// Save two slots for CDXUTSDKMesh diffuse and normal, 
// so start with the third slot, t2
Texture2D<float4> g_OffScreenBuffer : register( t2 );

//--------------------------------------------------------------------------------------
// shader input/output structure
//--------------------------------------------------------------------------------------
struct VS_QUAD_OUTPUT
{
    float4 Position     : SV_POSITION;
    float2 TextureUV    : TEXCOORD0;
};


//--------------------------------------------------------------------------------------
// Function:    FullScreenQuadVS
//
// Description: Vertex shader that generates a fullscreen quad with texcoords.
//              To use draw 3 vertices with primitive type triangle strip
//--------------------------------------------------------------------------------------
VS_QUAD_OUTPUT FullScreenQuadVS( uint id : SV_VertexID )
{
    VS_QUAD_OUTPUT Out = (VS_QUAD_OUTPUT)0;

    float2 vTexCoord = float2( (id << 1) & 2, id & 2 );

    // z = 1 below because we have inverted depth
    Out.Position = float4( vTexCoord * float2( 2.0f, -2.0f ) + float2( -1.0f, 1.0f), 1.0f, 1.0f );
    Out.TextureUV = vTexCoord;

    return Out;
}

//--------------------------------------------------------------------------------------
// Function:    FullScreenBlitPS
//
// Description: Copy input to output.
//--------------------------------------------------------------------------------------
float4 FullScreenBlitPS( VS_QUAD_OUTPUT i ) : SV_TARGET
{
    return g_OffScreenBuffer.SampleLevel(g_Sampler, i.TextureUV, 0);
}

