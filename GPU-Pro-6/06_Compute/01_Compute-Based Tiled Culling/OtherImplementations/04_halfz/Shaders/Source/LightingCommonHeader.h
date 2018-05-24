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
// File: LightingCommonHeader.h
//
// HLSL file for the ComputeBasedTiledCulling sample. Header file for lighting.
//--------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------------------

void DoLighting(in Buffer<float4> PointLightBufferCenterAndRadius, in Buffer<float4> PointLightBufferColor, in uint nLightIndex, in float3 vPosition, in float3 vNorm, in float3 vViewDir, out float3 LightColorDiffuseResult, out float3 LightColorSpecularResult)
{
    float4 CenterAndRadius = PointLightBufferCenterAndRadius[nLightIndex];

    float3 vToLight = CenterAndRadius.xyz - vPosition;
    float3 vLightDir = normalize(vToLight);
    float fLightDistance = length(vToLight);

    LightColorDiffuseResult = float3(0,0,0);
    LightColorSpecularResult = float3(0,0,0);

    float fRad = CenterAndRadius.w;
    if( fLightDistance < fRad )
    {
        float x = fLightDistance / fRad;
        // fake inverse squared falloff:
        // -(1/k)*(1-(k+1)/(1+k*x^2))
        // k=20: -(1/20)*(1 - 21/(1+20*x^2))
        float fFalloff = -0.05 + 1.05/(1+20*x*x);
        LightColorDiffuseResult = PointLightBufferColor[nLightIndex].rgb * saturate(dot(vLightDir, vNorm)) * fFalloff;

        float3 vHalfAngle = normalize( vViewDir + vLightDir );
        LightColorSpecularResult = PointLightBufferColor[nLightIndex].rgb * pow(saturate(dot(vHalfAngle, vNorm)), 8) * fFalloff;
    }
}

void DoSpotLighting(in Buffer<float4> SpotLightBufferCenterAndRadius, in Buffer<float4> SpotLightBufferColor, in Buffer<float4> SpotLightBufferSpotParams, in uint nLightIndex, in float3 vPosition, in float3 vNorm, in float3 vViewDir, out float3 LightColorDiffuseResult, out float3 LightColorSpecularResult)
{
    float4 BoundingSphereCenterAndRadius = SpotLightBufferCenterAndRadius[nLightIndex];
    float4 SpotParams = SpotLightBufferSpotParams[nLightIndex];

    // reconstruct z component of the light dir from x and y
    float3 SpotLightDir;
    SpotLightDir.xy = SpotParams.xy;
    SpotLightDir.z = sqrt(1 - SpotLightDir.x*SpotLightDir.x - SpotLightDir.y*SpotLightDir.y);

    // the sign bit for cone angle is used to store the sign for the z component of the light dir
    SpotLightDir.z = (SpotParams.z > 0) ? SpotLightDir.z : -SpotLightDir.z;

    // calculate the light position from the bounding sphere (we know the top of the cone is 
    // r_bounding_sphere units away from the bounding sphere center along the negated light direction)
    float3 LightPosition = BoundingSphereCenterAndRadius.xyz - BoundingSphereCenterAndRadius.w*SpotLightDir;

    float3 vToLight = LightPosition - vPosition;
    float3 vToLightNormalized = normalize(vToLight);
    float fLightDistance = length(vToLight);
    float fCosineOfCurrentConeAngle = dot(-vToLightNormalized, SpotLightDir);

    LightColorDiffuseResult = float3(0,0,0);
    LightColorSpecularResult = float3(0,0,0);

    float fRad = SpotParams.w;
    float fCosineOfConeAngle = (SpotParams.z > 0) ? SpotParams.z : -SpotParams.z;
    if( fLightDistance < fRad && fCosineOfCurrentConeAngle > fCosineOfConeAngle)
    {
        float fRadialAttenuation = (fCosineOfCurrentConeAngle - fCosineOfConeAngle) / (1.0 - fCosineOfConeAngle);
        fRadialAttenuation = fRadialAttenuation * fRadialAttenuation;

        float x = fLightDistance / fRad;
        // fake inverse squared falloff:
        // -(1/k)*(1-(k+1)/(1+k*x^2))
        // k=20: -(1/20)*(1 - 21/(1+20*x^2))
        float fFalloff = -0.05 + 1.05/(1+20*x*x);
        LightColorDiffuseResult = SpotLightBufferColor[nLightIndex].rgb * saturate(dot(vToLightNormalized,vNorm)) * fFalloff * fRadialAttenuation;

        float3 vHalfAngle = normalize( vViewDir + vToLightNormalized );
        LightColorSpecularResult = SpotLightBufferColor[nLightIndex].rgb * pow( saturate(dot( vHalfAngle, vNorm )), 8 ) * fFalloff * fRadialAttenuation;
    }
}
