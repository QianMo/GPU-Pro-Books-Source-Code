#define BE_RENDERABLE_INCLUDE_WORLD
#define BE_RENDERABLE_INCLUDE_PROJ

#include <Engine/Perspective.fx>
#include <Engine/Renderable.fx>
#include <Pipelines/LPR/Geometry.fx>

cbuffer SetupConstants
{
	float4 DiffuseColor
	<
		String UIName = "Diffuse";
		String UIWidget = "Color";
	> = float4(1.0f, 1.0f, 1.0f, 1.0f);

	float4 SpecularColor
	<
		String UIName = "Specular";
		String UIWidget = "Color";
	> = float4(0.1f, 0.1f, 0.1f, 1.0f);
	
	float Roughness
	<
		String UIName = "Roughness";
		String UIWidget = "Slider";
		float UIMin = 0.0f;
		float UIMax = 1.0f;
	> = 0.4f;

	float Reflectance
	<
		String UIName = "Reflectance";
		String UIWidget = "Slider";
		float UIMin = 0.0f;
		float UIMax = 1.0f;
	> = 0.0f;

	float Metalness
	<
		String UIName = "Metalness";
		String UIWidget = "Slider";
		float UIMin = 0.0f;
		float UIMax = 1.0f;
	> = 0.0f;

	float MetalFresnel
	<
		String UIName = "Metal Fresnel";
		String UIWidget = "Slider";
		float UIMin = 0.0f;
		float UIMax = 1.0f;
	> = 0.0f;

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

GBufferBinding PSGeometry(Pixel p)
{
	return BindGBuffer(
		MakeGeometry(p.NormalDepth.w, normalize(p.NormalDepth.xyz)),
		MakeDiffuse(DiffuseColor.xyz, Roughness),
		MakeSpecular(SpecularColor.xyz, SpecularColor.w, saturate(Reflectance + DiffuseColor.w), Metalness, MetalFresnel) );
}

technique11 Geometry <
	string PipelineStage = "GeometryPipelineStage";
	string RenderQueue = "DefaultRenderQueue";
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
	string IncludeEffect = "Prototypes/Shadow.fx";
> { }

technique11 <
	string IncludeEffect = "Prototypes/Feedback.fx";
> { }
