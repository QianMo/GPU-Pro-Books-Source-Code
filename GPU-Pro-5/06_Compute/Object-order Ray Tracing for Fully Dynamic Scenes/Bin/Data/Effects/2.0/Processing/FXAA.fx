#define BE_SCENE_SETUP

#define FXAA_PC 1
#define FXAA_LINEAR 1
#define FXAA_HLSL_5 1
#define FXAA_QUALITY__PRESET 15

#ifdef FXAA_3_8
	#include <External/Fxaa3_8.h>
#else
	#include <External/Fxaa3_11.h>
#endif

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
	float2 Center : TexCoord0;
	float4 LeftTopRightBottom : TexCoord1;
};

Pixel VSQuad(Vertex v)
{
	Pixel p;

	p.Position = v.Position;
	p.Center = 0.5f + float2(0.5f, -0.5f) * v.Position.xy;
	p.LeftTopRightBottom = p.Center.xyxy + float2(-0.5f, 0.5f).xxyy * DestinationResolution.zwzw;

	return p;
}

SamplerState LinearSampler
{
	Filter = MIN_MAG_MIP_LINEAR;
};

float4 PSFXAA(Pixel p) : SV_Target0
{
	FxaaTex inputTexture = { LinearSampler, LDRTexture };
#ifdef FXAA_3_8
	return FxaaPixelShader(p.Center, p.LeftTopRightBottom, inputTexture, DestinationResolution.zw, float2(2.0f, 0.5f).xxyy * DestinationResolution.zwzw);
#else
	return FxaaPixelShader(p.Center, p.LeftTopRightBottom, inputTexture, inputTexture, inputTexture,
		DestinationResolution.zw, 0.0f, 0.0f, 0.0f,
		0.75f, 0.166f, 0.0833f,
		0.0f, 0.0f, 0.0f, 0.0f);
#endif
}

technique11 Default <
	string PipelineStage = "ProcessingPipelineStage";
	bool DontFlip = true;
>
{
	pass <
		string Color0 = "FinalTarget";
		bool bClearColor0 = true;
	>
	{
		SetRasterizerState( NULL );
		SetDepthStencilState( NULL, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetVertexShader( CompileShader(vs_5_0, VSQuad()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader(ps_5_0, PSFXAA()) );
	}
}
