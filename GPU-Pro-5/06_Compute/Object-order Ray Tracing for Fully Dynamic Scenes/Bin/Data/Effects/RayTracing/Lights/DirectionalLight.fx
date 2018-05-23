/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "2.0/Lights/DirectionalLight.fx"
#include "Pipelines/Tracing/Ray.fx"
#include "Pipelines/Tracing/Scene.fx"
#include <Utility/Math.fx>

static const uint LightingGroupSize = 512;

[numthreads(LightingGroupSize,1,1)]
void CSMain(uint dispatchIdx : SV_DispatchThreadID, uniform bool bShadowed = true)
{
	if (dispatchIdx >= RaySet.RayCount)
		return;

	TracedGeometry geometry = TracedGeometryBuffer[dispatchIdx];

	if (geometry.Depth >= MissedRayDepth)
		return;

	RayDesc ray = RayDescriptionBuffer[dispatchIdx];

	float4 diffuseColor = ExtractTracedColor(geometry.Diffuse);
	float4 specularColor = ExtractTracedColor(geometry.Specular);
	
	float3 normal = normalize( ExtractTracedNormal(geometry.Normal) );
	float3 camDir = normalize(ray.Dir);

	float3 intensity = 1.0f;

	if (bShadowed)
	{
		float3 pos = ray.Orig + asfloat(geometry.Depth) * ray.Dir;

		float splitDepth = max3(abs(mul(float4(pos, 1.0f), Perspective.View).xyz));
		int splitIndex = (int) dot(splitDepth >= DirectionalShadow[0].ShadowSplitPlanes, 1.0f);

		if (0 <= splitIndex && splitIndex < 4)
		{
			float4 shadowCoord = mul(float4(pos + 0.1f * normal, 1.0f), DirectionalShadow[0].ShadowSplits[splitIndex].Proj);
			float2 shadowMapCoord = 0.5f + float2(0.5f, -0.5f) * shadowCoord.xy;

			if ( all( float4(shadowMapCoord.xy, 1.0f - shadowMapCoord.xy) >= 0.0f ) )
			{
				float shadowRange = DirectionalShadow[0].ShadowSplits[splitIndex].NearFar.y - DirectionalShadow[0].ShadowSplits[splitIndex].NearFar.x;

				float shadowDepth = DirectionalLightShadowMaps.SampleLevel(ShadowSampler, float3(shadowMapCoord, splitIndex), 0.0f).r;
				float shadowDeltaDepth = (shadowCoord.z - shadowDepth * shadowCoord.w) * shadowRange;

				intensity = saturate(1.0f - 50.0f * max(shadowDeltaDepth - 0.5f, 0.0f) );
			}
		}
	}

	float3 halfway = -normalize(camDir + DirectionalLight[0].Dir);
	float r0 = 1.0f - 0.7f * (1.0f - diffuseColor.w);
	float r3 = r0 + (1.0f - r0) * pow(1.0f + dot(halfway, camDir), 2.0f);

	// Angle fallof
	float cosAngle = dot(normal, -DirectionalLight[0].Dir);
	float negIntensity = saturate(0.5f - 0.35f * cosAngle);
	float3 posIntensity = saturate(cosAngle) * intensity;

	posIntensity = lerp(posIntensity, 0.5f * DirectionalLight[0].Color.w * DirectionalLight[0].SkyColor.xyz, r3);

	float3 diffuse = diffuseColor.xyz * lerp(posIntensity, DirectionalLight[0].SkyColor.xyz * negIntensity, DirectionalLight[0].Color.w);

	// Specular
	float3 specular = specularColor.xyz * intensity * pow( saturate( dot(normal, halfway) ) , 1024.0f * specularColor.a );
	
	float4 light = float4( (diffuse + specular) * DirectionalLight[0].Color.xyz, 0.0f );
	
	TracedLightUAV[dispatchIdx].Color = PackTracedLight(light);
}

[numthreads(1,1,1)]
void CSTracingGroup(uint dispatchIdx : SV_DispatchThreadID)
{
	GroupDispatchUAV.Store3(
			0,
			uint3( ceil_div(RaySet.RayCount, LightingGroupSize), 1, 1 )
		);
}

technique11 TracingDefault <
	string PipelineStage = "TraceLightingPipelineStage";
	string RenderQueue = "DefaultRenderQueue";
	bool EnableTracing = true;
>
{
	pass < string TracingCSGroupPass = "Group"; string LightType = "DirectionalLight"; bool Shadowed = false; >
	{
		// Workaround
		SetVertexShader( CompileShader(vs_5_0, VSMain()) );
		SetComputeShader( CompileShader(cs_5_0, CSMain(false)) );
	}

	pass < string TracingCSGroupPass = "Group"; string LightType = "DirectionalLight"; bool Shadowed = true; >
	{
		// Workaround
		SetVertexShader( CompileShader(vs_5_0, VSMain()) );
		SetComputeShader( CompileShader(cs_5_0, CSMain(true)) );
	}

	pass Group < bool Normal = false; >
	{
		SetComputeShader( CompileShader(cs_5_0, CSTracingGroup()) );
	}
}
