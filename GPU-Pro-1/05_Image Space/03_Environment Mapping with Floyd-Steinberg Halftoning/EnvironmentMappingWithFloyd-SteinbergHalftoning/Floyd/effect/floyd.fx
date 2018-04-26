#include "enginePool.fx"

Texture2D geometryViewportMap;
Texture2D brdfViewportMap;

Texture2D randomMap;
uint2 tileCorner = uint2(300, 196);
uint2 viewportSize = uint2(512, 512);

struct EnvironmentSample
{
	float4 screenPos : POSITION;
	float4 dir : DIRECTION;
};

void vsRandomSampleEnvironment()
{
}

[maxvertexcount(32)]
void gsRandomSampleEnvironment( uint primitiveId : SV_primitiveID,
							   inout PointStream<EnvironmentSample> sampleStream )
{
	uint2 targetPixel = uint2(primitiveId % 128, primitiveId / 128);
	targetPixel += tileCorner;

	float4 screenPos = float4(targetPixel.xy, 0, 0) / float4(viewportSize.xy, 1, 1) * float4(2, -2, 0, 0) + float4(-1, 1, 0, 1);
	float4 geo = geometryViewportMap.Load(int3(targetPixel, 0));
	if(geo.w == 0)
		return;
	screenPos.z = geo.w;
	float3 normal = geo.xyz;
	float3 tangent = normalize(cross(normal, float3(0.57, 0.57, 0.57)));
	float3 binormal = cross(normal, tangent);
	float3x3 tangentFrame = float3x3(tangent, binormal, normal);
	
	for(int i=0; i<32; i++)
	{
		EnvironmentSample sample;
		sample.screenPos = screenPos;

		float3 sam = randomMap.Load(uint3(
			targetPixel.x%64*8 + i%8, targetPixel.y%128*4 + i/8, 0));
		sample.dir.xyz = mul(sam, tangentFrame);
		sample.dir.w = 1;
		sampleStream.Append( sample );
	}
}

technique10 randomSampleEnvironment
{
    pass P0
	{
        SetVertexShader ( CompileShader( vs_4_0, vsRandomSampleEnvironment() ) );
        SetGeometryShader( 
	        ConstructGSWithSO( CompileShader( gs_4_0, gsRandomSampleEnvironment()),
				"POSITION.xyzw; DIRECTION.xyzw" ) 
				);
		SetRasterizerState( defaultRasterizer );
        SetPixelShader( NULL );
		SetDepthStencilState( noDepthTestCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

struct SampleContribution
{
	float4 pos : SV_POSITION;
	float4 color : COLOR;
};

SampleContribution vsEvaluateSamples(EnvironmentSample input)
{
	float3 normal = geometryViewportMap.Load(input.screenPos).xyz;

	SampleContribution output;
	float dist = input.screenPos.z;
	input.screenPos.z = 0;
	output.pos = input.screenPos;
	float4 worldPosFromEye = mul(input.screenPos, orientProjMatrixInverse);
	worldPosFromEye /= worldPosFromEye.w;
	float3 worldPos = eyePosition + normalize(worldPosFromEye.xyz) * dist;

	float3 rdir = normalize(reflect(worldPosFromEye.xyz, normal));

	output.color = float4(0, 0, 0, 1.0/32.0);
//	if(trace(worldPos, input.dir.xyz) > 900000.0)
		output.color = 
			float4(envMap.SampleLevel(linearSampler, input.dir.xyz, 0).xyz
			/ input.dir.w * 0.3 * float3(1, 1, 0)
			, 1.0/16.0);

	return output;
}

float4 psEvaluateSamples(SampleContribution input) : SV_TARGET
{
	return input.color;
}

Texture2D rawViewportMap;

technique10 evaluateSamples
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsEvaluateSamples() ) );
        SetGeometryShader( NULL	);
		SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psEvaluateSamples() ) );
		SetDepthStencilState( noDepthTestCompositor, 0);
		SetBlendState(additiveBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

float4 psAlphaNormalize(QuadOutput input) : SV_TARGET
{
	float4 r = rawViewportMap.Load(uint3(input.pos.xy, 0));
	if(r.w == 0)
		return envMap.Sample(linearSampler, input.viewDir);
	return r;
}

technique10 alphaNormalize
{
	pass P0
	{
        SetVertexShader ( CompileShader( vs_4_0, vsQuad() ) );
        SetGeometryShader( NULL );
		SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psAlphaNormalize() ) );
		SetDepthStencilState( noDepthTestCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
	}
}

float4 psBackgroundUnderlay(QuadOutput input) : SV_TARGET
{
	return envMap.Sample(linearSampler, input.viewDir) * 0.3;
};

technique10 backgroundUnderlay
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsQuad() ) );
        SetGeometryShader( NULL );
		SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psBackgroundUnderlay() ) );
		SetDepthStencilState( noDepthTestCompositor, 0);
		SetBlendState(underlayBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

#include "halftoning.fx"