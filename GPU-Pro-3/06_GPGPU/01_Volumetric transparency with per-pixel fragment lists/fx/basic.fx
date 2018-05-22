float4x4 modelMatrix;
float4x4 modelMatrixInverse;
float4x4 viewProjMatrix;
float4x4 modelViewProjMatrix;
float4x4 viewMatrix;
float4x4 viewDirMatrix;	//< from NDC to world space, minus eyeposition

float3 eyePos;
TextureCube envTexture;

Texture2D kdTexture;
float4 kdColor;

SamplerState linearSampler
{
	Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

/// Input assembler's output structure for the classic per-pixel lighting technique
struct IaosTrafo
{
    float4 pos			: POSITION;
    float3 normal		: NORMAL;
    float2 tex			: TEXCOORD;
};

/// Vertex shaders's output structure for the classic per-pixel lighting technique
struct VsosTrafo
{
    float4 pos			: SV_POSITION;
	float4 worldPos		: WORLDPOS;
    float3 normal		: NORMAL;
    float2 tex			: TEXCOORD;
};

/// Vertex shaders for the classic per-pixel lighting technique
VsosTrafo vsTrafo(IaosTrafo input)
{
	VsosTrafo output = (VsosTrafo)0;
	output.pos = mul(input.pos,
		modelViewProjMatrix);
	output.worldPos = mul(input.pos,
		modelMatrix);
	output.normal = mul(modelMatrixInverse,
		float4(input.normal.xyz, 0.0));
	output.tex = input.tex;
	return output;
}


/// Input assembler's output structure for rendering a full viewport quad
struct IaosQuad
{
	float4  pos			: POSITION;
	float2  tex			: TEXCOORD0;
};

/// Vertex shader's output structure for rendering a full viewport quad
struct VsosQuad
{
	float4 pos				: SV_POSITION;
	float2 tex				: TEXCOORD0;
	float3 viewDir			: TEXCOORD1;
};

/// Vertex shader's for rendering a full viewport quad, computing view direction
VsosQuad vsQuad(IaosQuad input)
{
	VsosQuad output = (VsosQuad)0;

	output.pos = input.pos;
    float4 hViewDir = mul(input.pos, viewDirMatrix);
    hViewDir /= hViewDir.w;
    output.viewDir = hViewDir.xyz;
    output.tex = input.tex;
    return output;
};