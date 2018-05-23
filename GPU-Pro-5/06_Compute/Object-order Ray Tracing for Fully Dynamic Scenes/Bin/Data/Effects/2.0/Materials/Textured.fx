#define BE_RENDERABLE_INCLUDE_WORLD
#define BE_RENDERABLE_INCLUDE_PROJ

#include <Engine/Perspective.fx>
#include <Engine/Renderable.fx>
#include <Pipelines/LPR/Geometry.fx>
#include <Utility/Math.fx>

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

	float2 TextureRepeat
	<
		String UIName = "Texture Repeat";
	> = float2(1.0f, 1.0f);

#ifdef VARMAP
	float2 VarianceRepeat
	<
		String UIName = "Variance Repeat";
	> = float2(1.0f, 1.0f);
#endif

#ifdef SECONDCOLOR
	float4 SecondColor
	<
		String UIName = "Second";
		String UIWidget = "Color";
	> = float4(1.0f, 1.0f, 1.0f, 1.0f);
#endif

	#hookinsert SetupConstants
}

#hookincl "Hooks/Transform.fx"
// NOTE: Include AFTER constant buffer to make sure constants are available
#hookincl ...

#ifndef NOCOLORMAP

	Texture2D DiffuseTexture
	<
		string UIName = "Diffuse";
	>;

	#define IF_COLORMAP(x) x
	#define IF_NOCOLORMAP(x)

#else

	#define IF_COLORMAP(x)
	#define IF_NOCOLORMAP(x) x

#endif

#ifdef VARMAP

	Texture2D VarianceTexture
	<
		string UIName = "Variance";
	>;

	#define IF_VARMAP(x) x
	#define IF_NOVARMAP(x)

#else

	#define IF_VARMAP(x)
	#define IF_NOVARMAP(x) x

#endif

#ifdef NORMALMAP

	float4 NormalBlend
	<
		String UIName = "Normal Blend";
	> = float4(0.0f, 0.0f, 0.0f, 1.0f);

	Texture2D NormalTexture
	<
		string UIName = "Normal";
		bool Raw = true;
	>;

	bool PremultipliedNormal
	<
		string UIName = "PremultipliedNormal";
		bool Raw = true;
	> = false;

	#define IF_NORMALMAP(x) x

#else
	#define IF_NORMALMAP(x)
#endif

#ifdef SPECULARMAP

	Texture2D SpecularTexture
	<
		string UIName = "Specular";
		bool Raw = true;
	>;

	#define IF_SPECULARMAP(x) x

#else
	#define IF_SPECULARMAP(x)
#endif

#if defined(SECONDCOLOR)
	#define NEED_WORLDPOS
#endif

struct Vertex
{
	float4 Position	: Position;
	float3 Normal	: Normal;
#ifndef GLOBAL_PLANE_COORDS
	float2 TexCoord	: TexCoord;
#endif

	IF_NORMALMAP(
		float3 Tangent	: Tangent;
	)
};

struct TransformedVertexData
{
	float4 NormalDepth	: TexCoord0;
	float2 TexCoord		: TexCoord1;
	
	IF_NORMALMAP(
		float4 Tangent	: TexCoord2;
	)

#ifdef NEED_WORLDPOS
	float3 WorldPos : TexCoord3;
#endif
};

struct TransformedVertex : TransformedVertexData
{
	float4 Position : Position;
};

struct Pixel : TransformedVertexData
{
	float4 Position : SV_Position;
};

Pixel VSMain(Vertex v)
{
	Pixel o;

	#hookcall transformState = Transform(v);

	o.Position = GetWVPPosition(transformState);
	o.NormalDepth.xyz = GetWorldNormal(transformState);
	o.NormalDepth.w = o.Position.w;

#ifdef NEED_WORLDPOS
	o.WorldPos = GetWorldPosition(transformState);
#endif

#ifdef GLOBAL_PLANE_COORDS
//	o.TexCoord = GetWorldPosition(transformState).xz * o.TexCoord.zw;
	o.TexCoord.xy = v.Position.xz * o.TexCoord.zw;
#else
	o.TexCoord.xy = v.TexCoord * TextureRepeat;
#endif

	IF_NORMALMAP(
		o.Tangent.xyz = mul(v.Tangent, (float3x3) World);
		// Extract handedness
		o.Tangent.w = sign(dot(v.Tangent, v.Tangent) - 0.5f);
	)

	return o;
}

SamplerState LinearSampler
{
	Filter = ANISOTROPIC;
	AddressU = WRAP;
	AddressV = WRAP;
};

float3 DecodeNormal(float3 n)
{
	n.xy = n.xy * 2.0f - 1.0f;
	return normalize(n);
}

GBufferBinding PSGeometry(Pixel p, uint primID : SV_PrimitiveID)
{
	float4 multiplier = DiffuseColor;

#ifdef SECONDCOLOR
	multiplier *= lerp( 1.0f, SecondColor, saturate(dot(normalize(p.NormalDepth.xyz), normalize(Perspective.CamPos - p.WorldPos))) );
#endif

	float2 texCoord = p.TexCoord.xy;
	float2 texCoordDDX = ddx(texCoord), texCoordDDY = ddy(texCoord);
	float3 normal = normalize(p.NormalDepth.xyz);

	float4 diffuse = multiplier;
	float3 debug = 0.0f;
	
	IF_NORMALMAP(
		float3 bitangent = normalize( cross(normal, p.Tangent.xyz) );
		float3 tangent = cross(bitangent, normal);
		
		// Texture might be flipped
		// ORDER: Don't flip before full frame has been computed!
		bitangent *= p.Tangent.w;
	)

	IF_COLORMAP(
		diffuse.xyz *= DiffuseTexture.Sample(LinearSampler, texCoord).xyz;
	)

	IF_VARMAP(
		float3 varColor = VarianceTexture.Sample(LinearSampler, texCoord * VarianceRepeat).xyz;
		diffuse.xyz = lerp(diffuse.xyz, multiplier * varColor, 0.7f * saturate(0.04f * p.NormalDepth.w));
//		diffuse.xyz = lerp(diffuse.xyz, sqrt(diffuse.xyz * varColor), saturate(max3(diffuse.xyz) * 5));
	)

#ifdef TRI_COLOR
	float4 triColor = frac(float4(0.53f, 0.73f, 0.91f, 0.107f) * (primID + 1));
	diffuse.xyz = triColor.xyz;
#endif

	float4 specular = SpecularColor;
	
	IF_SPECULARMAP(
		specular.xyz *= SpecularTexture.Sample(LinearSampler, texCoord).xyz;

		// HACK?: Reflection
		diffuse.w *= dot(specular.xyz, 1.0f / 3.0f);
	)

	IF_NORMALMAP(
		float4 normalSample = NormalTexture.Sample(LinearSampler, texCoord);
		
		if (PremultipliedNormal)
			normalSample.xyz /= normalSample.w;

		float3 mapNormal = mul(
				DecodeNormal( normalSample.xyz ),
				float3x3(tangent, bitangent, normal)
			);
		
		float3 bentNormal = normalize( lerp(normal, mapNormal, NormalBlend.w) );

#ifndef STRAIGHT_NORMALS
		if (dot(bentNormal, normal) > 0.0f)
			normal = bentNormal;
#endif

		IF_NOCOLORMAP(
			diffuse.xyz = lerp(NormalBlend.xyz, diffuse.xyz, normalSample.w);
			specular *= pow(saturate(2 * normalSample.w), 2);
		)
	)

	if (isnan(normal.x))
		normal = float3(0.0f, 1.0f, 0.0f);

	return BindGBuffer(
		MakeGeometry(p.NormalDepth.w, normal),
		MakeDiffuse(diffuse.xyz, Roughness),
		MakeSpecular(specular.xyz, specular.w, saturate(Reflectance + diffuse.w), Metalness, MetalFresnel) );
}

technique11 Geometry <
	string PipelineStage = "GeometryPipelineStage";
	string RenderQueue = "DefaultRenderQueue";
>
{
	pass
	{
		SetVertexShader( CompileShader(vs_5_0, VSMain()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader(ps_5_0, PSGeometry()) );
	}
}

technique11 <
	string IncludeEffect = "Prototypes/Shadow.fx";
	string IncludeHooks[] = #hookarray;
> { }

technique11 <
	string IncludeEffect = "Prototypes/Feedback.fx";
	string IncludeHooks[] = #hookarray;
> { }
