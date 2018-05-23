/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#define BE_SCENE_SETUP
#define BE_PERSPECTIVE_SETUP
#define BE_TRACING_SETUP
#define BE_VOXEL_REP_SETUP

#include <Engine/Perspective.fx>
#include <Pipelines/LPR/Scene.fx>
#include <Pipelines/LPR/Geometry.fx>

#include <Pipelines/Tracing/Ray.fx>
#include <Pipelines/Tracing/VoxelRep.fx>
#include <Pipelines/Tracing/RaySet.fx>

#include <Engine/Pipe.fx>

#include <Utility/Math.fx>
#include <Utility/Bits.fx>

// #define PRIMARY
// #define REFRACTION
// #define RAYS_EVERYWHERE
 #define MORE_REFLECTION

Texture2D<uint> RayCountTexture : RayCountTarget
<
	string TargetType = "Permanent";
	string Format = "R32U";
	uint MipLevels = 0;
	bool AutoGenMips = false;
>;
Texture2D<uint> RayOffsetTexture : RayOffsetTarget
<
	string TargetType = "Permanent";
	string Format = "R32U";
	uint MipLevels = 0;
	bool AutoGenMips = false;
>;
Texture2D<uint> RayIndexTexture : RayIndexTarget
<
	string TargetType = "Permanent";
	string Format = "R32U";
>;
float2 RayIndexTextureResolution : RayIndexTargetResolution;

RWStructuredBuffer<RayDesc> RayQueueUAV : register(u4) : RayQueueUAV;

struct Vertex
{
	float4 Position : Position;
};

struct Pixel
{
	float4 Position : SV_Position;
	float2 TexCoord : TexCoord0;
	float3 CamDir	: TexCoord1;
};

Pixel VSQuad(Vertex v)
{
	Pixel p;

	p.Position = v.Position;
	p.TexCoord = 0.5f + float2(0.5f, -0.5f) * v.Position.xy;

	// Compute per-pixel camera direction
	p.CamDir = v.Position.xyw * float3(Perspective.ProjInv[0][0], Perspective.ProjInv[1][1], 1.0f);
	p.CamDir = mul(p.CamDir, (float3x3) Perspective.ViewInv);

	return p;
}

SamplerState DefaultSampler
{
	Filter = MIN_MAG_MIP_POINT;
};

SamplerState LinearSampler
{
	Filter = MIN_MAG_MIP_LINEAR;
};

static const uint RayGenUnordered = 0;
static const uint RayGenCoherentAlloc = 1;
static const uint RayGenCoherentWrite = 2;

uint4 PSRayGen(Pixel p, uniform uint tech) : SV_Target0
{
	// Sample g-buffer
	float4 eyeGeometry = SceneGeometryTexture[p.Position.xy];
	GBufferSpecular gbSpecular = ExtractSpecular( SceneSpecularTexture[p.Position.xy] );
	float eyeDepth = ExtractDepth(eyeGeometry);

	// Check, if reflecting
#ifdef RAYS_EVERYWHERE
	if (eyeDepth < 1000.0f)
#else
	if (gbSpecular.FresnelR > 0.001f)
#endif
	{
		float3 camDir = normalize(p.CamDir);

		// Reconstruct position & normal
		float3 worldPos = Perspective.CamPos.xyz + p.CamDir * eyeDepth;
		float3 worldNormal = normalize( ExtractNormal(eyeGeometry) );

		// Compute secondary ray direction
		float3 worldReflection = reflect(camDir, worldNormal);

#ifdef PRIMARY
		// Test: generate primary rays
		worldPos = Perspective.CamPos.xyz + 0.005f * camDir;
		worldReflection = camDir;
#endif
#ifdef REFRACTION
		float n = 0.94f;
		float cosI = -dot(camDir, worldNormal);
		float sinT2 = n * n * (1.0f - cosI * cosI);
		sinT2 = saturate(sinT2); // hack
		float cosT = sqrt(1.0f - sinT2);
		worldReflection = normalize( n * p.CamDir + (n * cosI - cosT) * worldNormal );
		worldPos -= 0.09f * worldNormal;
#endif

		// Enqueue ray
		if (tech == RayGenCoherentAlloc)
			return 1;
		else
		{
			// Always count number of rays
			uint rayIdx = RayQueueUAV.IncrementCounter();

			// Coherent ray generation: place ray at compacted z-order position (2D prefix sum scan)
			if (tech == RayGenCoherentWrite)
				rayIdx = RayOffsetTexture[(uint2) p.Position.xy];

			RayDesc rayDesc = { worldPos, worldReflection };
			RayQueueUAV[rayIdx] = rayDesc;

			// Store ray index for final gather
			return rayIdx;
		}
	}

	if (tech == RayGenCoherentAlloc)
		return 0;
	else
		return -1;
}

StructuredBuffer<RayDesc> RayDescriptionBuffer : RayDescriptions;
StructuredBuffer<TracedGeometry> TracedGeometryBuffer : TracedGeometry; // Hit points, currently unused
StructuredBuffer<TracedLight> TracedLightBuffer : TracedLight;

StructuredBuffer<RayDebug> DebugRays : DebugRays; // Currently unused
Texture3D<uint> DebugGrid : DebugGrid; // Currently unused

float4 PSRayGather(Pixel p) : SV_Target0
{
	// Get ray index
	uint rayIdx = RayIndexTexture[ (uint2) (p.TexCoord * RayIndexTextureResolution) ];

	// Check if reflecting
	if (rayIdx != -1)
	{
		// Sample g-buffer
		float3 normal = ExtractNormal( SceneGeometryTexture[p.Position.xy] );
		GBufferDiffuse gbDiffuse = ExtractDiffuse( SceneDiffuseTexture[p.Position.xy] );
		GBufferSpecular gbSpecular = ExtractSpecular( SceneSpecularTexture[p.Position.xy] );

		RayDesc ray = RayDescriptionBuffer[rayIdx];
		float3 camDir = normalize(Perspective.CamPos.xyz - ray.Orig);
		
		// Fresnel/Reflectivity
		float3 r0 = gbSpecular.FresnelR;
#ifdef MORE_REFLECTION
		r0 = 1.0f - 0.8f * (1.0f - gbSpecular.FresnelR);
#endif
		float fres = pow(1.0f - abs(dot(normal, camDir)), 5);
#ifdef PRIMARY
		r0 = fres = 1.0f;
#endif
#ifdef REFRACTION
		fres = 1.0f - fres;
#endif
		float3 refl = lerp(r0, 1.0f, fres);
		float reflA = dot(refl, 0.3333f);
		float3 metalColor = lerp(gbSpecular.Color * 0.2f, 1.0f, fres);
		
		// Blend reflection
		float3 light = ExtractTracedLight(TracedLightBuffer[rayIdx].Color).xyz;
		light = lerp(light, metalColor * light, gbSpecular.Metalness);
		return float4(refl * light, reflA);
	}
	
	return 0.0f;
}

// Simple 2D prefix sum scan to obtain compacted z-order layout
Texture2D<uint> CountTexture : RayCountTexture;
Texture2D<uint> OffsetTexture : RayOffsetTexture;

static const uint ReduceStep = 2;
static const uint ReduceStepMask = ReduceStep - 1;

uint4 PSRayCount(Pixel p) : SV_Target0
{
	uint2 baseIdx = ReduceStep * (uint2) p.Position;

	uint count = 0;

	for (uint j = 0; j < ReduceStep; ++j)
		for (uint i = 0; i < ReduceStep; ++i)
			count += CountTexture[baseIdx + uint2(i, j)];

	return count;
}

uint4 PSRayOffset(Pixel p) : SV_Target0
{
	uint2 offsetIdx = (uint2) p.Position & ReduceStepMask;
	uint2 baseIdx = (uint2) p.Position & ~ReduceStepMask; 

	uint offset = 0;

	if (offsetIdx.x || offsetIdx.y)
		offset += CountTexture[baseIdx + uint2(0, 0)];
	if (offsetIdx.y)
		offset += CountTexture[baseIdx + uint2(1, 0)];
	if (offsetIdx.x && offsetIdx.y)
		offset += CountTexture[baseIdx + uint2(0, 1)];

	return offset + OffsetTexture[(uint2) p.Position / ReduceStep];
}

/// Additive blend state.
BlendState AdditiveBlendState
{
	BlendEnable[0] = true;
	SrcBlend[0] = One;
	DestBlend[0] = Inv_Src_Alpha;
};

technique11 RayGather
{
	pass <
			string Color0 = "SceneTarget";
			bool bClearColorOnce0 = true;
			bool bKeepColor0 = true;
		>
	{
		SetRasterizerState( NULL );
		SetDepthStencilState( NULL, 0 );
		SetBlendState( AdditiveBlendState, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetVertexShader( CompileShader(vs_5_0, VSQuad()) );
		SetPixelShader( CompileShader(ps_5_0, PSRayGather()) );
	}
}

technique11 RayGen
{
	pass <
		string Color0 = "RayIndexTarget";
		bool bKeepColor0 = true;
	>
	{
		SetRasterizerState( NULL );
		SetDepthStencilState( NULL, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetVertexShader( CompileShader(vs_5_0, VSQuad()) );
		SetPixelShader( CompileShader(ps_5_0, PSRayGen(RayGenUnordered)) );
	}
}

technique11 RayGenCohAlloc
{
	pass <
		string Color0 = "RayCountTarget";
		bool bKeepColor0 = true;
	>
	{
		SetRasterizerState( NULL );
		SetDepthStencilState( NULL, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetVertexShader( CompileShader(vs_5_0, VSQuad()) );
		SetPixelShader( CompileShader(ps_5_0, PSRayGen(RayGenCoherentAlloc)) );
	}
}

technique11 RayGenCohWrite
{
	pass <
		string Color0 = "RayIndexTarget";
		bool bKeepColor0 = true;
	>
	{
		SetRasterizerState( NULL );
		SetDepthStencilState( NULL, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetVertexShader( CompileShader(vs_5_0, VSQuad()) );
		SetPixelShader( CompileShader(ps_5_0, PSRayGen(RayGenCoherentWrite)) );
	}
}

technique11 RayCount
{
	pass
	{
		SetRasterizerState( NULL );
		SetDepthStencilState( NULL, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetVertexShader( CompileShader(vs_5_0, VSQuad()) );
		SetPixelShader( CompileShader(ps_5_0, PSRayCount()) );
	}
}

technique11 RayOffset
{
	pass
	{
		SetRasterizerState( NULL );
		SetDepthStencilState( NULL, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetVertexShader( CompileShader(vs_5_0, VSQuad()) );
		SetPixelShader( CompileShader(ps_5_0, PSRayOffset()) );
	}
}
