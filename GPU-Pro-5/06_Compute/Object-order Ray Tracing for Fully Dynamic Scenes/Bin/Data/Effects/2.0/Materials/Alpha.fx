#define BE_RENDERABLE_INCLUDE_WORLD
#define BE_RENDERABLE_INCLUDE_PROJ

#include <Engine/Perspective.fx>
#include <Engine/Renderable.fx>

cbuffer SetupConstants
{
	float4 DiffuseColor
	<
		String UIName = "Diffuse";
		String UIWidget = "Color";
	> = float4(1.0f, 1.0f, 1.0f, 1.0f);
	
}

struct Vertex
{
	float4 Position	: Position;
	float3 Normal	: Normal;
};

struct Pixel
{
	float4 Position		: SV_Position;
	float4 NormalDepth	: TexCoord0;
};

Pixel VSMain(Vertex v)
{
	Pixel o;
	
	o.Position = mul(v.Position, WorldViewProj);
	o.NormalDepth.xyz = normalize( mul((float3x3) WorldInverse, v.Normal) );
	o.NormalDepth.w = o.Position.w;
	
	return o;
}

float4 PSGeometry(Pixel p) : SV_Target0
{
	return DiffuseColor;
}

technique11 Geometry <
	string PipelineStage = "DefaultPipelineStage";
	string RenderQueue = "AlphaRenderQueue";
>
{
	pass
	{
		SetVertexShader( CompileShader(vs_4_0, VSMain()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader(ps_4_0, PSGeometry()) );
	}
}

technique11 <
	string IncludeEffect = "Prototypes/Feedback.fx";
> { }
