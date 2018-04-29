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

#ifndef COMMON_PARTICLE_FX_H
#define COMMON_PARTICLE_FX_H

#include "GBuffer.hlsl"
#include "AVSM_Resolve.hlsl"
#include "Rendering.hlsl"
#include "ConstantBuffers.hlsl"

//--------------------------------------------------------------------------------------

//////////////////////////////////////////////
// Resource Views
//////////////////////////////////////////////
RWStructuredBuffer<AVSMData>  gAVSMStructBufUAV;
StructuredBuffer<AVSMData>    gAVSMStructBufSRV;
Texture2D                     gParticleOpacityNoiseTex;
       
bool RaySphereIntersection(out float2 interCoeff, in float3 sphereCenter, in float sphereRadius, in float3 rayOrigin, in float3 rayNormDir)
{
	float3 dst = rayOrigin - sphereCenter;
	float b = dot(dst, rayNormDir);
	float c = dot(dst, dst) - (sphereRadius * sphereRadius);
	float d = b * b - c;
	interCoeff = -b.xx + sqrt(d) * float2(-1,1);
	return d > 0;
}

// compute particles thickness assuming their billboards
// in screen space are maximal sections of a sphere.
float ParticleThickness(float2 ray)
{
	return saturate(1 - sqrt(2.0f) * length(ray));
}

// We model particles as small spheres and we intersect them with light rays
// Entry and exit points are used to defined an AVSM segment
bool IntersectDynamicParticle(in  DynamicParticlePSIn input,
                              out float3 entry,
                              out float3 exit,
                              out float  transmittance)
{
    bool   res;
    float2 linearCoeff;    
    float3 normRayDir     = normalize(input.ViewPos);
    float3 particleCenter = input.ViewCenter;
    float  particleSize = input.UVS.z;
    float  particleRadius = mScale * particleSize / 2.0f;
                
    [flatten]if (RaySphereIntersection(linearCoeff, 
                              particleCenter, 
                              particleRadius, 
                              float3(0, 0, 0), 
                              normRayDir)){  
        // compute entry and exit points along the ray direction
	    entry = linearCoeff.xxx * normRayDir;  
	    exit  = linearCoeff.yyy * normRayDir;
	    
	    // compute normalized opacity
	    float segLength = (exit.z - entry.z) / (2.0f * particleRadius); 
	    // Compute density based indirectly on distance from center
	    float densityTimesSegLength = pow(segLength, 4.0f);
	    // per particle and global opacity multipliers
        float opacity = input.Opacity * mParticleOpacity;	
        // compute transmittance
	    transmittance = exp(-opacity * densityTimesSegLength);          	    	    
	    res = true;  
    } 
    else 
    {
        entry = 0;
        exit  = 0;
        transmittance = 1;        
        res = false;
    }     

    return res;
}                   

float GetDynamicParticleAlpha(DynamicParticlePSIn Input)
{    
    float3 entry, exit;
    float  transmittance;
    IntersectDynamicParticle(Input, entry, exit, transmittance);
    
    return saturate(transmittance);
}

//--------------------------------------------------------------------------------------
DynamicParticlePSIn DynamicParticlesShading_VS(
    float4 inPosition	: POSITION,
    float3 inUV			: TEXCOORD0,
    float  inOpacity	: TEXCOORD1
)
{
    DynamicParticlePSIn	Out;
    float size		= inUV.z * mParticleSize;

    // Make screen-facing
    float4 position;
    float2 offset	= inUV.xy - 0.5f.xx;
    position.xyz	= inPosition.xyz + size * (offset.xxx * mEyeRight.xyz + offset.yyy * mEyeUp.xyz);  
    position.w		= 1.0;

    float4 projectedPosition = mul( position, mParticleWorldViewProj ); 
    
    Out.Position    = projectedPosition;
    
    Out.ObjPos      = position.xyz;
    Out.ViewPos 	= mul( position, mParticleWorldView ).xyz;
    Out.ViewCenter	= mul( float4(inPosition.xyz, 1.0f), mParticleWorldView).xyz;
    Out.UVS			= float3(inUV.xy, size);
    Out.Opacity		= inOpacity;

    return Out;
}

SurfaceData ConstructSurfaceData(float3 PosView, float3 Normal)
{
    SurfaceData Surface;
    Surface.positionView = PosView;
    Surface.positionViewDX = ddx(Surface.positionView);
    Surface.positionViewDY = ddy(Surface.positionView);
    Surface.normal = Normal;
    Surface.albedo = float4(0,0,0,1);
    Surface.lightSpaceZ = mul(float4(Surface.positionView.xyz, 1.0f), mCameraViewToLightProj).z;
    Surface.lightTexCoord = ProjectIntoLightTexCoord(Surface.positionView.xyz);
    Surface.lightTexCoordDX = ddx(Surface.lightTexCoord);
    Surface.lightTexCoordDY = ddy(Surface.lightTexCoord); 

    return Surface;
}

float ShadowContrib(SurfaceData LitSurface, DynamicParticlePSIn Input)
{
    float2 lightTexCoord = ProjectIntoAvsmLightTexCoord(LitSurface.positionView.xyz);
    float receiverDepth = mul(float4(LitSurface.positionView.xyz, 1.0f), mCameraViewToAvsmLightView).z;     
    
    return VolumeSample(mUI.volumeShadowMethod, lightTexCoord, receiverDepth);  
}

float4 DynamicParticlesShading_PS(DynamicParticlePSIn Input) : SV_Target
{
    float3 entry, exit;	
	float  shadowTerm = 1.0f;
	float  segmentTransmittance = 1.0f;
    [flatten]if (IntersectDynamicParticle(Input, entry, exit, segmentTransmittance)) {
	    float2 lightTexCoord = ProjectIntoLightTexCoord(entry);	    
	    float receiver = mul(float4(entry, 1.0f), mCameraViewToLightProj).z;  
			
        SurfaceData LitSurface = ConstructSurfaceData(entry, 0.0f.xxx);	
        if (mUI.enableVolumeShadowLookup) {
            shadowTerm = ShadowContrib(LitSurface, Input);
        }
    }

	float depthDiff = 1.0f;
	float3 diffuse = 1.0f;

	float3 LightContrib = float3(0.8,0.8,1.0) * diffuse;
    [flatten]if (mbSoftParticles) {		
		// Calcualte the difference in the depths		
		SurfaceData SurfData = ComputeSurfaceDataFromGBuffer(int2(Input.Position.xy));
		
		// If the depth read-in is zero, there was nothing rendered at that pixel.
		// In such a case, we set depthDiff to 1.0f.
		if (SurfData.positionView.z == 0) {
			depthDiff = 1.0f;
		} else {
			depthDiff = (SurfData.positionView.z - Input.ViewPos.z) / mSoftParticlesSaturationDepth;
			depthDiff = smoothstep(0.0f, 1.0f, depthDiff);
		}
	}
  
    float3 Color = LightContrib * shadowTerm;// * gParticleOpacityNoiseTex.Sample(gDiffuseSampler, Input.UVS.xy).xyz;
    return float4(Color, depthDiff * (1.0f - segmentTransmittance));   
}

[earlydepthstencil]
void ParticleAVSMCapture_PS(DynamicParticlePSIn Input)
{
    float3 entry, exit;
    float  segmentTransmittance;
	if (IntersectDynamicParticle(Input, entry, exit, segmentTransmittance)) {
	
        // Allocate a new node
        // (If we're running out of memory we simply drop this fragment
        uint newNodeAddress;       
        
        if (LT_AllocSegmentNode(newNodeAddress)) {
            // Fill node
            ListTexSegmentNode node;            
            node.depth[0] = entry.z;
            node.depth[1] = exit.z;
            node.trans    = segmentTransmittance;
            node.sortKey  = Input.ViewPos.z;
            
	        // Get fragment viewport coordinates
            int2 screenAddress = int2(Input.Position.xy);            
            
            // Insert node!
            LT_InsertFirstSegmentNode(screenAddress, newNodeAddress, node);            
        }           
	} 
}


uint ConvertPixelPosToLinearTiledAddress(uint2 pixelPos)
{
    const uint tileSize = 8;
    const uint tileSizeSq = tileSize * tileSize;
    uint2 tileId = pixelPos / tileSize;
    uint2 tilePos = pixelPos % tileSize;
    uint address = tileId.y * ((uint)mShadowMapSize/tileSize) * tileSizeSq + 
                   tileId.x * tileSizeSq + 
                   tilePos.y * tileSize + 
                   tilePos.x;

    return address;
}

void ParticleAVSMSinglePassInsert_PS(DynamicParticlePSIn Input)
{
    float  segmentTransmittance;    
    float3 entry, exit;    
    if (IntersectDynamicParticle(Input, entry, exit, segmentTransmittance)) 
    {	    
        uint fragmentCount;

        // Get fragment viewport coordinates
        int2 screenAddress = int2(Input.Position.xy);      	  
         
        //DeviceMemoryBarrier();      	  
        
        bool cond = false;
        [allow_uav_condition]while (!cond) 
        {        
            InterlockedCompareExchange(gListTexFirstSegmentNodeAddressUAV[screenAddress], 0, 1, fragmentCount);    
            
            if (1 == fragmentCount) 
            {
                cond = true;                

                // Read AVSM from RW-UAV (RWStructuredBuffer)
                uint address = ConvertPixelPosToLinearTiledAddress((uint2)Input.Position.xy);
                
                //DeviceMemoryBarrier();
                
                AVSMData avsmData = gAVSMStructBufUAV[address];
                             
                // Insert segment into AVSM
                float segmentDepth[2] = {entry.z, exit.z};
                InsertSegmentAVSM(segmentDepth, segmentTransmittance, avsmData);
                                
                // Write AVSM back to memory
                gAVSMStructBufUAV[address] = avsmData;         

                //DeviceMemoryBarrier();

                InterlockedCompareStore(gListTexFirstSegmentNodeAddressUAV[screenAddress], 1, 0);               
            } 
        }
    }
}

void AVSMClearStructuredBuf_PS(FullScreenTriangleVSOut Input)
{
    // Compute linearized address of this pixel
    uint2 screenAddress = uint2(Input.positionViewport.xy);
    uint address = screenAddress.y * (uint)(mShadowMapSize) + screenAddress.x; 

    // Initialize AVSM data (clear)
    gAVSMStructBufUAV[address] = AVSMGetEmptyNode();
}

AVSMData_PSOut AVSMConvertSUAVtoTex2D_PS(FullScreenTriangleVSOut Input)
{
    // Read AVSM from RW-UAV (RWStructuredBuffer)
    uint address = ConvertPixelPosToLinearTiledAddress((uint2)Input.positionViewport.xy);
    AVSMData avsmData = gAVSMStructBufSRV[address]; 

    // Store final AVSM data into a 2D texture
    return avsmData;
}


#endif // COMMON_PARTICLE_FX_H
