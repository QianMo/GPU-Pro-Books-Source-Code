#include "..\\..\\MaterialAging\\MaterialPacking.h"
#include "ColorPacking.hlsl"

//-------------------------------------------------------------------------
// Constant Buffers
//-------------------------------------------------------------------------

cbuffer cbViewProjection : register( b0 )
{
	row_major matrix g_View = matrix(1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1);
	row_major matrix g_Projection = matrix(1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1);
	float3 g_Eye = float3(0.0, 0.0, 0.0);
};

cbuffer cbWorld : register( b1 )
{
	row_major matrix g_World = matrix(1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1);
	row_major matrix g_InvWorld = matrix(1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1);
};


//-------------------------------------------------------------------------
// Shader Resources
//-------------------------------------------------------------------------
Texture2D diffuseTexture : register( t0 );
Texture2D specularTexture : register( t1 );
Texture2D normalTexture : register( t2 );
Texture2D heightTexture : register( t3 );

SamplerState samplerLinear : register( s0 );
SamplerState samplerPoint : register( s1 );


//-------------------------------------------------------------------------
// Effect Properties
//-------------------------------------------------------------------------
float3 LightPosition = float3(278.0f, 530.7f, 279.5f);
float4 LightColor = float4(1, 1, 0.8, 1);

float4 Diffuse = float4(0.25, 0.5, 0.75, 1);
float SpecularValue = 0.0;

float4 OriginalMaterialProperty = float4(0, 0, 0, 0);		// used during initialization of the atlas
float4 OriginalMaterialProperty2 = float4(0, 0, 0, 0);		// it will contain the materials defined in the presets

bool AgedScene = false;		// same shader for rendering the aged and the non-aged scene

bool UseColorMap = false;
bool UseSpecularMap = false;
bool UseNormalMap = false;
bool UseHeightMap = false;

float AtlasSize = 4096;				// size in number of pixels

// Tessellation parameters.
float g_MinTessFactor = 1;
float g_MaxTessFactor = 6;
float g_MinTessDist = 1000;
float g_MaxTessDist = 100;
float g_DisplacementStrength = 1;

//-------------------------------------------------------------------------
// Data passed between shader stages
//-------------------------------------------------------------------------

struct VERTEX {
	float4 Position : POSITION0;
	float3 Normal : NORMAL0;
	float2 TexCoord : TEXCOORD0;
};

struct HS_CONSTANT_DATA {
	float Edges[3]	: SV_TessFactor;
	float Inside[1]	: SV_InsideTessFactor;
	float3 FaceNormal : FACE_NORMAL;
	float4 Tangent[3] : TANGENT;
};

struct PS_INPUT {
	float4 Position : SV_POSITION;
	float3 WorldPos : TEXCOORD0;
	float3 Normal : NORMAL0;
	float4 Tangent : TANGENT0;
	float2 TexCoord : TEXCOORD1;
	float3 FaceNormal : TEXCOORD2;
};

struct PS_OUTPUT_ToAtlas {
	float4 Color : SV_Target0;
	float4 GeomNormal : SV_TARGET1;
	float4 GeomTangentFrame : SV_TARGET2;
	float SurfHeight : SV_TARGET3;
	float4 SurfNormal : SV_TARGET4;
	float4 Material : SV_TARGET5;
	float4 Material2 : SV_TARGET6;
	float4 SplatSize : SV_TARGET7;
};

//--------------------------------------------------------------------------------------
// Vertex Shader - transforms to world space
//--------------------------------------------------------------------------------------

VERTEX VS(VERTEX vertex) 
{
	vertex.Position = mul(vertex.Position, g_World);
	vertex.Normal = mul(float4(vertex.Normal, 1), g_World).xyz;
	return vertex;
}

//--------------------------------------------------------------------------------------
// Hull Shader - Passes tesselation factors to tesselator
//--------------------------------------------------------------------------------------

float dtf(float3 pos) { // distance-dependent tessellation factors
	return lerp(g_MinTessFactor, g_MaxTessFactor, 
		smoothstep(g_MinTessDist, g_MaxTessDist, distance(g_Eye, pos)));
}

//	Lengyel, Eric. “Computing Tangent Space Basis Vectors for an Arbitrary Mesh”.
//	Terathon Software 3D Graphics Library, 2001. 
//	http://www.terathon.com/code/tangent.html
void computeTangentFrame(in VERTEX vertex[3], out float4 tangent[3], out float3 faceNormal)
{
	float3 v0 = vertex[0].Position.xyz;
	float3 v1 = vertex[1].Position.xyz;
	float3 v2 = vertex[2].Position.xyz;
	
	float2 w0 = vertex[0].TexCoord;
	float2 w1 = vertex[1].TexCoord;
	float2 w2 = vertex[2].TexCoord;
	
	float x1 = v1.x - v0.x;
	float x2 = v2.x - v0.x;
	float y1 = v1.y - v0.y;
	float y2 = v2.y - v0.y;
	float z1 = v1.z - v0.z;
	float z2 = v2.z - v0.z;
	
	float s0 = w1.x - w0.x;
	float s1 = w2.x - w0.x;
	float t0 = w1.y - w0.y;
	float t1 = w2.y - w0.y;
	
	float r = 1.0 / (s0 * t1 - s1 * t0);
	float3 tan0 = float3((t1 * x1 - t0 * x2) * r, (t1 * y1 - t0 * y2) * r, (t1 * z1 - t0 * z2) * r);
	float3 tan1 = float3((s0 * x2 - s1 * x1) * r, (s0 * y2 - s1 * y1) * r, (s0 * z2 - s1 * z1) * r);
	
	float3 n0 = vertex[0].Normal;
	float3 n1 = vertex[1].Normal;
	float3 n2 = vertex[2].Normal;
	float3 t = tan0;
        
	// Gram-Schmidt orthogonalize
	tangent[0].xyz = normalize(t - n0 * dot(n0, t));
	tangent[1].xyz = normalize(t - n1 * dot(n1, t));
	tangent[2].xyz = normalize(t - n2 * dot(n2, t));
	
	// Calculate handedness
	tangent[0].w = (dot(cross(n0, t), tan1) < 0.0) ? -1.0 : 1.0;
	tangent[1].w = (dot(cross(n1, t), tan1) < 0.0) ? -1.0 : 1.0;
	tangent[2].w = (dot(cross(n2, t), tan1) < 0.0) ? -1.0 : 1.0;
	
	// calculate face normal
	faceNormal = normalize(cross(v2 - v0, v1 - v0));
}

HS_CONSTANT_DATA ConstantHS( InputPatch<VERTEX, 3> ip)
{    
	HS_CONSTANT_DATA output;
	// compute tessellation factors for each vertex
	float3 tess = float3(dtf(ip[0].Position.xyz), dtf(ip[1].Position.xyz), dtf(ip[2].Position.xyz));
	output.Edges[0] = max(tess.y, tess.z);
	output.Edges[1] = max(tess.x, tess.z);
	output.Edges[2] = max(tess.x, tess.y);
	output.Inside[0] = max(tess.x, output.Edges[0]);	
	
	// optional: do frustum culling here

	// compute tangents (could be precomputed)
	computeTangentFrame(ip, output.Tangent, output.FaceNormal);
	return output;
}

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("ConstantHS")]
[maxtessfactor(63.0)]
VERTEX HS( InputPatch<VERTEX, 3> p, uint i : SV_OutputControlPointID)
{
	VERTEX output;
	output.Position = p[i].Position;
	output.TexCoord = p[i].TexCoord;
	output.Normal = p[i].Normal;
	return output;
}

//--------------------------------------------------------------------------------------
// Domain Shader - uses barycentric interpolation to compute the attributes
//--------------------------------------------------------------------------------------
[domain("tri")]
PS_INPUT DS( HS_CONSTANT_DATA input,  
					float3 UV : SV_DomainLocation,
					const OutputPatch<VERTEX, 3> bezpatch )
{
	PS_INPUT output = (PS_INPUT)0;
	output.WorldPos = bezpatch[0].Position.xyz * UV.x + bezpatch[1].Position.xyz * UV.y + bezpatch[2].Position.xyz * UV.z;	
	output.Normal =	  normalize(bezpatch[0].Normal) * UV.x + normalize(bezpatch[1].Normal)   * UV.y + normalize(bezpatch[2].Normal)   * UV.z;
	output.TexCoord = bezpatch[0].TexCoord * UV.x + bezpatch[1].TexCoord * UV.y + bezpatch[2].TexCoord * UV.z;
	float displacement = (heightTexture.SampleLevel(samplerPoint, output.TexCoord, 0).x	- 0.5) * g_DisplacementStrength;
	output.WorldPos += normalize(output.Normal) * displacement;
	output.Position = mul(mul(float4(output.WorldPos, 1), g_View), g_Projection);

	output.FaceNormal = input.FaceNormal;
	output.Tangent = input.Tangent[0] * UV.x + input.Tangent[1] * UV.y + input.Tangent[2] * UV.z;
	
	return output;
}


//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------

// this is hardwired and works for all non-metal materials
// optional: store these for each material (maybe in another altas texture)
#define specularCoefficient float3(0.04, 0.04, 0.04) 

float3 blinnPhong(float3 Nn, float3 Vn, float3 Ln, float3 diffuseCoefficient, float shininess)
{
	float3 Hn = normalize(Ln + Vn);

	// diffuse
	float3 diffuse = (1.0 / 3.14159) * diffuseCoefficient;

	// specular
	float3 schlick = specularCoefficient + (1.0 - specularCoefficient) * pow(1.0 - dot(Vn, Hn), 5.0);
	float3 specular = (shininess + 2) * 0.125 * pow(saturate(dot(Nn, Hn)), shininess) * schlick;

	return diffuse + specular;
}

float4 PS( PS_INPUT input) : SV_Target0
{
	// we use a simple directional light source
	float3 Ln = normalize(LightPosition);
	float3 Nn = normalize(input.Normal);
	float3 Vn = normalize(g_Eye - input.WorldPos);

	// normal mapping
	float3 Tn = normalize(input.Tangent.xyz);
	float3 Bn = normalize(cross(Nn, Tn)) * input.Tangent.w;
	Tn = cross(Nn, -Bn); // this is important since the original tangent is a face tangent and the normal is not (they not orthogonal)
	float3x3 TBN = transpose(float3x3(Tn, Bn, Nn));
	float3 mapNormalWS = normalize((normalTexture.Sample( samplerLinear, input.TexCoord).xyz) * 2.0 - 1.0);	
	Nn = UseNormalMap ? mapNormalWS : float3(0,0,1);
	Ln = normalize(mul(Ln, TBN));
	Vn = normalize(mul(Vn, TBN));

	// we are using the same shader for rendering aged and original scene
	// se here is a small difference
	float4 packedColorValue;
	if(AgedScene)
	{
		packedColorValue = diffuseTexture.Sample( samplerLinear, input.TexCoord);
	}
	else
	{
		packedColorValue.xyz = diffuseTexture.Sample( samplerLinear, input.TexCoord).rgb;
		packedColorValue.w = specularTexture.Sample( samplerLinear, input.TexCoord).x;
	}

	// material parameters
	float3 diffuseColor;
	float shininess;
	UnpackColor(packedColorValue, diffuseColor, shininess);

	return float4(blinnPhong(Nn, Vn, Ln, diffuseColor, shininess) * LightColor.xyz * saturate(dot(Nn,Ln)), 1);
}

//--------------------------------------------------------------------------------------
// Geometry Shader  (in to atlas pass)
//--------------------------------------------------------------------------------------

[maxvertexcount(3)]
void GS( triangle VERTEX input[3], inout TriangleStream<PS_INPUT> TriStream )
{
	float4 tangents[3];
	float3 faceNormal;
	computeTangentFrame(input, tangents, faceNormal);

	PS_INPUT output;
	output.Position = float4(input[0].TexCoord.x * 2.0 - 1.0, 1 - 2.0 * input[0].TexCoord.y, 1, 1);
	output.WorldPos = input[0].Position.xyz;
	output.Normal = input[0].Normal;
	output.Tangent = tangents[0];
	output.TexCoord = input[0].TexCoord;
	output.FaceNormal = faceNormal;
	TriStream.Append( output );
	
	output.Position = float4(input[1].TexCoord.x * 2.0 - 1.0, 1 - 2.0 * input[1].TexCoord.y, 1, 1);
	output.WorldPos = input[1].Position.xyz;
	output.Normal = input[1].Normal;
	output.Tangent = tangents[1];
	output.TexCoord = input[1].TexCoord;
	TriStream.Append( output );
	
	output.Position = float4(input[2].TexCoord.x * 2.0 - 1.0, 1 - 2.0 * input[2].TexCoord.y, 1, 1);
	output.WorldPos = input[2].Position.xyz;
	output.Normal = input[2].Normal;
	output.Tangent = tangents[2];
	output.TexCoord = input[2].TexCoord;
	TriStream.Append( output );
}


/*
	Here we define the initial material properties
	The biggest contribution comes from the "MaterialProperty" which is defined in the model part
	there can be also influences of the geometric surface properties like colors, normal or hight, ...
*/
MaterialProperty InitMaterial(float3 diffuseColor, float specularValue, float3 normal, float height)
{
	MaterialProperty mat = (MaterialProperty) 0;

	// optional: add influence of other surface properties

	mat.Water = OriginalMaterialProperty.x;
	mat.Dirt = OriginalMaterialProperty.y;
	mat.Metal = OriginalMaterialProperty.z;
	mat.Wood = OriginalMaterialProperty.w;
	mat.Organic = OriginalMaterialProperty2.x;
	mat.Rust = OriginalMaterialProperty2.y;
	mat.Stone = OriginalMaterialProperty2.z;

	mat.Dummy = 0; // free slot

	return mat;
}

PS_OUTPUT_ToAtlas PS_ToAtlas( PS_INPUT input)
{
	PS_OUTPUT_ToAtlas output = (PS_OUTPUT_ToAtlas) 0;

	// diffuse
	float3 diffuseTexValue = diffuseTexture.SampleLevel( samplerPoint, input.TexCoord, 0).xyz;
	float3 diffuse = UseColorMap? diffuseTexValue : Diffuse.rgb;

	// specular
	float4 specTexValue = specularTexture.SampleLevel( samplerPoint, input.TexCoord, 0);
	float specularValue = UseSpecularMap ? 0.333 * (specTexValue.x + specTexValue.y + specTexValue.z) : SpecularValue;

	// smooth geometry normal
	float3 geometryNn = normalize(input.Normal);

	// tangent frame
	float3 Tn = normalize(input.Tangent.xyz);
	float3 Bn = normalize(cross(geometryNn, Tn)) * input.Tangent.w;
	Tn = cross(geometryNn, -Bn);

	// inverse TBN matrix to transform from tangent space to world space
	float3x3 invTBN = float3x3(Tn, Bn, geometryNn);

	// detailed surface normal
	float3 surfNormal;
	if(UseNormalMap)
	{
		// transform the normal map vector to world space
		surfNormal = mul( normalize(normalTexture.SampleLevel( samplerPoint, input.TexCoord, 0).xyz * 2.0 - 1.0), invTBN);
	}
	else
	{
		// if no normal map is availabe use the geometry normal
		surfNormal = geometryNn;
	}

	// height
	// if there is no height map we assume average height
	float height = UseHeightMap ? heightTexture.SampleLevel( samplerPoint, input.TexCoord, 0).x : 0.5;

	// material
	MaterialProperty mat = InitMaterial(diffuse, specularValue, surfNormal, height);

	// texture space stretching
	float2 deltaPos;
	deltaPos.x = length(ddx(input.WorldPos) * AtlasSize);	// to make dx independet from atlas resolution
	deltaPos.y = length(ddy(input.WorldPos) * AtlasSize);	// to make dx independet from atlas resolution
	deltaPos /= 1000;	// scaled to be able to see whats up in debug output

	// output
	output.Color = float4(diffuse, specularValue); // to use this for shading use UnpackColor to get diffuse color coefficient and shininess
	output.GeomNormal = float4(geometryNn * 0.5 + 0.5, 1);		// vectors are stored in [0,1] range
	output.GeomTangentFrame = input.Tangent * 0.5 + 0.5;
	output.SurfHeight = height;
	output.SurfNormal = float4(surfNormal * 0.5 + 0.5, 1);
	output.Material = float4(mat.Water, mat.Dirt, mat.Metal, mat.Wood);
	output.Material2 = float4(mat.Organic, mat.Rust, mat.Stone, mat.Dummy);
	output.SplatSize = float4(deltaPos,0, 1);
	return output;
}


//--------------------------------------------------------------------------------------
technique11 ShaderModel5
{
	pass Default
	{
		SetVertexShader( CompileShader( vs_5_0, VS() ) );
		SetHullShader( CompileShader( hs_5_0, HS() ) );
		SetDomainShader( CompileShader( ds_5_0, DS() ) );
		SetGeometryShader( 0 );
		SetPixelShader( CompileShader( ps_5_0, PS() ) );
	}

	pass ToAtlas
	{
		SetVertexShader( CompileShader( vs_5_0, VS() ) );
		SetHullShader( 0 );
		SetDomainShader( 0 );
		SetGeometryShader( CompileShader( gs_5_0, GS() ) );
		SetPixelShader( CompileShader( ps_5_0, PS_ToAtlas() ) );
	}
}