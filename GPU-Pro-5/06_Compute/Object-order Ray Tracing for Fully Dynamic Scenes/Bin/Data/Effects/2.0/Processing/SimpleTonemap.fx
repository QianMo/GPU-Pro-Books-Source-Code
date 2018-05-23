#define BE_SCENE_SETUP

#include <Engine/Pipe.fx>
#include <Pipelines/Scene.fx>
#include <Engine/Perspective.fx>

float4 DestinationResolution : DestinationResolution;

struct Vertex
{
	float4 Position : Position;
};

struct Pixel
{
	float4 Position : SV_Position;
	float2 TexCoord : TexCoord0;
};

Pixel VSQuad(Vertex v)
{
	Pixel p;

	p.Position = v.Position;
	p.TexCoord = 0.5f + float2(0.5f, -0.5f) * v.Position.xy;

	return p;
}

SamplerState LinearSampler
{
	Filter = MIN_MAG_MIP_LINEAR;
};

float4 NoiseFromPixelPosition(uint2 position, uint2 resolution, uint seed)
{
	uint idx = position.x + position.y * resolution.x;
	uint area = position.x * position.y;

	const uint4 idcs = seed + idx * uint4(41, 29, 53, 43);
	const uint4 areas = seed + area  * uint4(23, 59, 47, 37);

	return float4((idcs ^ areas) % 661) / 330.5f - 1.0f;
}

float3 NaughtyHDR(float3 x)
{
	float ShoulderStrength = 0.22f; // A
	float LinearStrength = 0.30f; // B
	float LinearAngle = 0.10f; // C
	float ToeStrength = 0.20f; // D
	float ToeNumerator = 0.01f; // E
	float ToeDenominator = 0.30f; // F
//	float LinearWhitePointValue = 11.2f;
	float ToeAngle = ToeNumerator / ToeDenominator;

	return (x * (ShoulderStrength * x + LinearAngle * LinearStrength) + ToeStrength * ToeNumerator)
		/ (x * (ShoulderStrength * x + LinearStrength) + ToeStrength * ToeDenominator)
		- ToeAngle;
}

float4 PSTonemap(Pixel p) : SV_Target0
{
	float4 hdrColor = SceneTexture.Sample(LinearSampler, p.TexCoord);

//	float hdrLum = dot( hdrColor, float3(0.2126f, 0.7152f, 0.0722f) );
//	hdrColor.xyz = hdrColor.xyz / (1.0f + hdrLum);
//	hdrColor.xyz = 1 - exp2(-hdrColor.xyz);

	float L = dot(hdrColor.xyz, float3(0.2126f, 0.7152f, 0.0722f));
	float TL = NaughtyHDR(L).x / NaughtyHDR(11.2f).x;

	hdrColor.xyz *= TL / L;
//	hdrColor.xyz = NaughtyHDR(hdrColor.xyz) / NaughtyHDR(11.2f);
	
	hdrColor.xyz += NoiseFromPixelPosition(p.Position.xy, DestinationResolution.xy, Perspective.Time * 731).xyz / 500.0f;

#ifdef BE_LDR_PROCESSING
	hdrColor.a = sqrt( dot(hdrColor.rgb, float3(0.299f, 0.587f, 0.114f)) );
#endif

	return hdrColor;
}

technique11 Default <
	string PipelineStage = "ProcessingPipelineStage";
	bool DontFlip = true;
>
{
	pass <
#ifndef BE_LDR_PROCESSING
		string Color0 = "FinalTarget";
#else
		string Color0 = "LDRTarget";
#endif
		bool bClearColor0 = true;
	>
	{
		SetRasterizerState( NULL );
		SetDepthStencilState( NULL, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetVertexShader( CompileShader(vs_4_0, VSQuad()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader(ps_4_0, PSTonemap()) );
	}
}
