#define BE_RENDERABLE_INCLUDE_WORLD
#define BE_RENDERABLE_INCLUDE_PROJ

#include <Engine/Perspective.fx>
#include <Engine/Renderable.fx>

cbuffer SetupConstants
{
	float4 GroundColor
	<
		String UIName = "Ground";
	> = float4(0.2f, 0.35f, 0.25f, -1.0f) * float4(0.2f, 0.35f, 0.25f, 1.0f);

	float4 HazeColor
	<
		String UIName = "Haze";
	> = float4(1.8f, 1.29784f, 0.9526176f, 0.f) * float4(1.8f, 1.29784f, 0.9526176f, 1.f);

	float4 SkyColor
	<
		String UIName = "Sky";
	> = float4(0.6272f, 0.8031999f, 1.1232f, 0.9f) * float4(0.6272f, 0.8031999f, 1.1232f, 1.0f);
}

struct Vertex
{
	float4 Position	: Position;
};

struct Pixel
{
	float4 Position		: SV_Position;
	float3 Direction	: TexCoord0;
};

Pixel VSMain(Vertex v)
{
	Pixel o;
	
	o.Position = mul(v.Position, WorldViewProj);
	o.Position.z = (1.0f - 4.8e-7f) * o.Position.w;
	o.Direction = mul(v.Position.xyz, (float3x3) World);
	
	return o;
}

#include "Atmospherics/SkyColor.fx"

float4 PSMain(Pixel p) : SV_Target0
{
	float3 direction = normalize(p.Direction);

	float3 color = EvaluateSkyColor(direction, SkyColor, HazeColor, true, GroundColor);

	return float4(color, 1.0f);
}

/// CCW rasterizer state disabling multi-sampling.
RasterizerState CCWRasterizerState
{
	FrontCounterClockwise = true;
};

technique11 Default <
	string PipelineStage = "DefaultPipelineStage";
	string RenderQueue = "DefaultRenderQueue";
>
{
	pass
	{
		SetRasterizerState(CCWRasterizerState);

		SetVertexShader( CompileShader(vs_4_0, VSMain()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader(ps_4_0, PSMain()) );
	}
}
