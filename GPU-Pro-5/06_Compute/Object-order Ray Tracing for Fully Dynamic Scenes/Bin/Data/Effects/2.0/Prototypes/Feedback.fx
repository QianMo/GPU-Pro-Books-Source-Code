#define BE_RENDERABLE_INCLUDE_PROJ
#define BE_RENDERABLE_INCLUDE_ID

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

float4 VSObjectIDs(Vertex v) : SV_Position
{
	return GetWVPPosition( #hookcall Transform(TransformHookPositionOnly(v.Position)) );
}

uint4 PSObjectIDs(float4 p : SV_Position) : SV_Target0
{
	return ObjectID;
}

float4 PSObjectIDColor(float4 p : SV_Position) : SV_Target0
{
	return frac(ObjectID / 3.3f);
}

technique11 ObjectIDs <
	string PipelineStage = "ObjectIDPipelineStage";
	string RenderQueue = "DefaultRenderQueue";
>
{
	pass
	{
		SetVertexShader( CompileShader(vs_4_0, VSObjectIDs()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader(ps_4_0, PSObjectIDs()) );
	}
}
/*
technique11 ObjectIDsColor <
	string PipelineStage = "DefaultPipelineStage";
	string RenderQueue = "DefaultRenderQueue";
>
{
	pass
	{
		SetVertexShader( CompileShader(vs_4_0, VSObjectIDs()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader(ps_4_0, PSObjectIDColor()) );
	}
}
*/