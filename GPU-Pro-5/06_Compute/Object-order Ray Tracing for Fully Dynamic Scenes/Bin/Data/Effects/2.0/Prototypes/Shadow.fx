#define BE_RENDERABLE_INCLUDE_PROJ

#include <Engine/Perspective.fx>
#include <Engine/Renderable.fx>

cbuffer SetupConstants
{
	#hookinsert SetupConstants
}

#hookincl "Hooks/Transform.fx"
#hookincl ...

struct Vertex
{
	float4 Position	: Position;
};

struct DepthPixel
{
	float4 Position		: SV_Position;
	float Depth			: TexCoord0;
};

DepthPixel VSDepth(Vertex v)
{
	#hookcall transformState = Transform(TransformHookPositionOnly(v.Position));

	DepthPixel o;
	o.Position = GetWVPPosition(transformState);
	o.Depth = o.Position.w;
	return o;
}

float4 PSDepth(DepthPixel p) : SV_Target0
{
	return p.Depth;
}

technique11 Shadow <
	string PipelineStage = "ShadowPipelineStage";
	string RenderQueue = "DefaultRenderQueue";
>
{
	pass
	{
		SetVertexShader( CompileShader(vs_4_0, VSDepth()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader(ps_4_0, PSDepth()) );
	}
}
