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
// File: Geometry.cpp
//
// Classes for geometric primitives and collision and visibility testing
//--------------------------------------------------------------------------------------
#include "..\\DXUT\\Core\\DXUT.h"
#include "Geometry.h"

using namespace DirectX;

//--------------------------------------------------------------------------------------
// Helper function to normalize a plane
//--------------------------------------------------------------------------------------
static void NormalizePlane( XMFLOAT4* pPlaneEquation )
{
    float mag;
    
    mag = sqrtf( pPlaneEquation->x * pPlaneEquation->x + 
                 pPlaneEquation->y * pPlaneEquation->y + 
                 pPlaneEquation->z * pPlaneEquation->z );
    
    pPlaneEquation->x = pPlaneEquation->x / mag;
    pPlaneEquation->y = pPlaneEquation->y / mag;
    pPlaneEquation->z = pPlaneEquation->z / mag;
    pPlaneEquation->w = pPlaneEquation->w / mag;
}


//--------------------------------------------------------------------------------------
// Extract all 6 plane equations from frustum denoted by supplied matrix
//--------------------------------------------------------------------------------------
void ExtractPlanesFromFrustum( XMFLOAT4* pPlaneEquation, const XMMATRIX* pMatrix, bool bNormalize )
{
    XMFLOAT4X4 TempMat;
    XMStoreFloat4x4( &TempMat, *pMatrix);

    // Left clipping plane
    pPlaneEquation[0].x = TempMat._14 + TempMat._11;
    pPlaneEquation[0].y = TempMat._24 + TempMat._21;
    pPlaneEquation[0].z = TempMat._34 + TempMat._31;
    pPlaneEquation[0].w = TempMat._44 + TempMat._41;
    
    // Right clipping plane
    pPlaneEquation[1].x = TempMat._14 - TempMat._11;
    pPlaneEquation[1].y = TempMat._24 - TempMat._21;
    pPlaneEquation[1].z = TempMat._34 - TempMat._31;
    pPlaneEquation[1].w = TempMat._44 - TempMat._41;
    
    // Top clipping plane
    pPlaneEquation[2].x = TempMat._14 - TempMat._12;
    pPlaneEquation[2].y = TempMat._24 - TempMat._22;
    pPlaneEquation[2].z = TempMat._34 - TempMat._32;
    pPlaneEquation[2].w = TempMat._44 - TempMat._42;
    
    // Bottom clipping plane
    pPlaneEquation[3].x = TempMat._14 + TempMat._12;
    pPlaneEquation[3].y = TempMat._24 + TempMat._22;
    pPlaneEquation[3].z = TempMat._34 + TempMat._32;
    pPlaneEquation[3].w = TempMat._44 + TempMat._42;
    
    // Near clipping plane
    pPlaneEquation[4].x = TempMat._13;
    pPlaneEquation[4].y = TempMat._23;
    pPlaneEquation[4].z = TempMat._33;
    pPlaneEquation[4].w = TempMat._43;
    
    // Far clipping plane
    pPlaneEquation[5].x = TempMat._14 - TempMat._13;
    pPlaneEquation[5].y = TempMat._24 - TempMat._23;
    pPlaneEquation[5].z = TempMat._34 - TempMat._33;
    pPlaneEquation[5].w = TempMat._44 - TempMat._43;
    
    // Normalize the plane equations, if requested
    if ( bNormalize )
    {
        NormalizePlane( &pPlaneEquation[0] );
        NormalizePlane( &pPlaneEquation[1] );
        NormalizePlane( &pPlaneEquation[2] );
        NormalizePlane( &pPlaneEquation[3] );
        NormalizePlane( &pPlaneEquation[4] );
        NormalizePlane( &pPlaneEquation[5] );
    }
}
