#include "enginePool.fx"

Texture2DArray<float> shadowMapArray;

struct Spotlight
{
	float4 position;
	float4 direction;
	float4 radianceParameters;
	float4x4 lightViewProjMatrix;
};

cbuffer spotlightBuffer
{
	uint4 nLights;
	Spotlight spotlights[512];
}

struct ShadowMapVertexInput
{
    float4 pos		: POSITION;
	uint instanceId	: SV_InstanceID;
};

struct ShadowMapGeometryInput
{
    float4 worldPos		: POSITION;
	uint instanceId		: INSTANCEID;
};

struct ShadowMapPixelInput
{
    float4 pos			: SV_POSITION;
    float3 worldPosMinusLightPos		: TEXCOORD0;
	uint rtIndex		: SV_RenderTargetArrayIndex;
};

ShadowMapGeometryInput
	vsShadowMap(ShadowMapVertexInput input)
{
	ShadowMapGeometryInput output = (ShadowMapGeometryInput)0;
	output.worldPos = mul(input.pos, modelMatrix);
	output.instanceId = input.instanceId;

	return output;
}

[maxvertexcount(3)]
void gsShadowMap( triangle ShadowMapGeometryInput input[3], inout TriangleStream<ShadowMapPixelInput> stream)
{
	uint iLight = input[0].instanceId;
	ShadowMapPixelInput output;
	output.rtIndex = iLight;
	output.pos = mul(input[0].worldPos, spotlights[iLight].lightViewProjMatrix);
	output.worldPosMinusLightPos = input[0].worldPos.xyz - spotlights[iLight].position.xyz;
	stream.Append(output);

	output.pos = mul(input[1].worldPos, spotlights[iLight].lightViewProjMatrix);
	output.worldPosMinusLightPos = input[1].worldPos.xyz - spotlights[iLight].position.xyz;
	stream.Append(output);

	output.pos = mul(input[2].worldPos, spotlights[iLight].lightViewProjMatrix);
	output.worldPosMinusLightPos = input[2].worldPos.xyz - spotlights[iLight].position.xyz;
	stream.Append(output);
	
	stream.RestartStrip();
}

float4 psShadowMap(ShadowMapPixelInput input) : SV_TARGET
{
	return length(input.worldPosMinusLightPos);
}

technique10 toShadowMap
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsShadowMap() ) );
        SetGeometryShader( CompileShader( gs_4_0, gsShadowMap() ) );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psShadowMap() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

[maxvertexcount(3)]
void gsDirectionalShadowMap( triangle ShadowMapGeometryInput input[3], inout TriangleStream<ShadowMapPixelInput> stream)
{
	uint iLight = input[0].instanceId;
	ShadowMapPixelInput output;
	output.rtIndex = iLight;
	output.pos = mul(input[0].worldPos, spotlights[iLight].lightViewProjMatrix);
	output.worldPosMinusLightPos = dot(input[0].worldPos.xyz, spotlights[iLight].direction.xyz);
	stream.Append(output);

	output.pos = mul(input[1].worldPos, spotlights[iLight].lightViewProjMatrix);
	output.worldPosMinusLightPos = dot(input[1].worldPos.xyz, spotlights[iLight].direction.xyz);
	stream.Append(output);

	output.pos = mul(input[2].worldPos, spotlights[iLight].lightViewProjMatrix);
	output.worldPosMinusLightPos = dot(input[2].worldPos.xyz, spotlights[iLight].direction.xyz);
	stream.Append(output);
	
	stream.RestartStrip();
}

float4 psDirectionalShadowMap(ShadowMapPixelInput input) : SV_TARGET
{
	return input.worldPosMinusLightPos.x;
}

technique10 toDirectionalShadowMap
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsShadowMap() ) );
        SetGeometryShader( CompileShader( gs_4_0, gsDirectionalShadowMap() ) );
        SetRasterizerState( backfaceRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psDirectionalShadowMap() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

float4 psShadowed(TrafoOutput input) : SV_TARGET
{
	float3 lighting = 0.05;
	uint nSpotlights = nLights.x;
	for(int il=0; il<nSpotlights; il++)
	{
		float4 lightScreenPos = mul(input.worldPos, spotlights[il].lightViewProjMatrix);
		lightScreenPos /= lightScreenPos.w;
		lightScreenPos.xy *= float2(0.5, -0.5);
		lightScreenPos.xy += float2(0.5, 0.5);
//		return lightScreenPos.xyxy;
		float shadowDist = shadowMapArray.SampleLevel(clampSampler, float3(lightScreenPos.xy, il), 0);

		float3 lightDiff = spotlights[il].position.xyz - input.worldPos;
		float lightDist = length(lightDiff);
		if(lightDist < shadowDist + 0.01)
		{
			float3 lightDir = lightDiff / lightDist;
			lighting += spotlights[il].radianceParameters.xyz * 
			 max(0, dot(input.normal, lightDir)) * 
			 pow(
				max(0, dot(-lightDir,spotlights[il].direction)),
				spotlights[il].radianceParameters.w)
				/dot(lightDiff, lightDiff);
		}
	 }
	 return float4(lighting * kdMap.Sample(linearSampler, input.tex), 1);
}

technique10 shadowed
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsTrafo() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psShadowed() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

float4 psEnvironmentLighted(TrafoOutput input) : SV_TARGET
{
	input.normal = normalize(input.normal);
	float3 viewDir = normalize(input.worldPos - eyePosition);

	float3 reflDir = reflect(viewDir, input.normal);
	float3 lighting = 0.0;
	uint nSpotlights = nLights.x;
	for(int il=0; il<nSpotlights; il++)
	{
		float4 lightScreenPos = mul(input.worldPos, spotlights[il].lightViewProjMatrix);
		lightScreenPos /= lightScreenPos.w;
		lightScreenPos.xy *= float2(0.5, -0.5);
		lightScreenPos.xy += float2(0.5, 0.5);
		float shadowDist = shadowMapArray.SampleLevel(clampSampler, float3(lightScreenPos.xy, il), 0);

		float lightDist = dot(spotlights[il].direction.xyz, input.worldPos);
		if(lightDist < shadowDist - 0.5 * dot(spotlights[il].direction.xyz, input.normal))
		{
			lighting += 
				spotlights[il].radianceParameters.xyz *
				pow(saturate(-dot(spotlights[il].direction.xyz, reflDir)), 60) * 5.1;
//				saturate(-dot(spotlights[il].direction.xyz, input.normal)) * 0.8 * 6;
		}
	 }
	 return float4(lighting /* * float3(1, 0.7, 0.5)/** kdMap.Sample(linearSampler, input.tex)*/, 1);
}

technique10 environmentLighted
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsTrafo() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psEnvironmentLighted() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

float4 psShowLights(QuadOutput input) : SV_TARGET
{
	float4 red = envMap.Sample(linearSampler, input.viewDir.zyx * float3(1,1,-1));
	input.viewDir = normalize(input.viewDir);
	uint nSpotlights = nLights.x;
	for(int il=0; il<nSpotlights; il++)
//		red.r += pow(saturate(-dot(input.viewDir, spotlights[il].direction.xyz)), 1800);
		if(-dot(input.viewDir, spotlights[il].direction.xyz) > 0.99999)
		{
			red.gb = 0;
			red.r = 100000;
		}

	return red;
};

technique10 showLights
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsQuad() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psShowLights() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

Texture2D importanceMap;

float4 psShowSampling(QuadOutput input) : SV_TARGET
{
//	input.tex.y *= 2;
	float4 red = importanceMap.Sample(linearSampler, input.tex) * sin(input.tex.y * 3.14);// * 0.15;
//	return red;
	input.tex.xy *= float2(6.28, 3.14);
//	float3 viewDir = float3(cos(input.tex.x) * sin(input.tex.y), cos(input.tex.y), -sin(input.tex.x) * sin(input.tex.y));

	uint nSpotlights = nLights.x;
	for(int il=0; il<nSpotlights; il++)
	{
		float2 phitheta = 
			float2(
				-atan2(spotlights[il].direction.z, spotlights[il].direction.x) + 3.14,
				3.14-acos(spotlights[il].direction.y));

//		red.r += pow(saturate(-dot(input.viewDir, spotlights[il].direction.xyz)), 1800);
//		if(-dot(viewDir, spotlights[il].direction.xyz) > 0.9999)
		if(length(phitheta - input.tex.xy) < 0.03)
		{
			red.gb = 0;
			red.r = 100000;
		}
	}

	return red;
};

technique10 showSampling
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsQuad() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psShowSampling() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}