#include "enginePool.fx"

struct Bone
{
	float4 orientation;
	float4 dualTranslation;
};

cbuffer boneTransforms
{
//	float2x4 bones[128];
	Bone bones[128];
};

struct SkinningInput
{
    float4 pos			: POSITION;
    float3 normal		: NORMAL;
    float2 tex			: TEXCOORD0;
	float4 blendWeights	: BLENDWEIGHT;
	uint4  blendIndices	: BLENDINDICES;
};

struct SkinningOutput
{
    float4 pos			: SV_POSITION;
    float3 normal		: TEXCOORD2;
    float2 tex			: TEXCOORD0;
    float4 worldPos		: TEXCOORD1;
	float4 color		: COLOR;
};

float4 quatMult(float4 a, float4 b)
{
	return float4(a.x * b.x - dot(a.yzw, b.yzw), cross(a.yzw, b.yzw) + a.x * b.yzw + b.x * a.yzw);
}

SkinningOutput
	vsSkinningDqs(SkinningInput input)
{
	SkinningOutput output = (SkinningOutput)0;

	input.blendWeights.w = 1 - dot(input.blendWeights.xyz, float3(1, 1, 1));

	float2x4 qe0 = float2x4(bones[input.blendIndices.x].orientation, bones[input.blendIndices.x].dualTranslation);
	float2x4 qe1 = float2x4(bones[input.blendIndices.y].orientation, bones[input.blendIndices.y].dualTranslation);
	float2x4 qe2 = float2x4(bones[input.blendIndices.z].orientation, bones[input.blendIndices.z].dualTranslation);
	float2x4 qe3 = float2x4(bones[input.blendIndices.w].orientation, bones[input.blendIndices.w].dualTranslation);

	float3 podality = float3(dot(qe0[0], qe1[0]), dot(qe0[0], qe2[0]), dot(qe0[0], qe3[0]));
	input.blendWeights.yzw *= (podality >= 0)?1:-1;
	
	float2x4 qe = input.blendWeights.x * qe0;
	qe += input.blendWeights.y * qe1;
	qe += input.blendWeights.z * qe2;
	qe += input.blendWeights.w * qe3;

	float len = length(qe[0]);
	qe /= len;

	float3 blendedPos =
		input.pos.xyz 
		+ 2 * cross(qe[0].yzw, 
									cross(qe[0].yzw, input.pos.xyz) 
								  + qe[0].x * input.pos.xyz);
	float3 trans = 2.0*(qe[0].x*qe[1].yzw - qe[1].x*qe[0].yzw + cross(qe[0].yzw, qe[1].yzw));
	blendedPos += trans;

	float3 inpNormal = input.normal.xyz;
	float3 normal = inpNormal + 2.0*cross(qe[0].yzw, cross(qe[0].yzw, inpNormal) + qe[0].x*inpNormal);

	output.pos = mul(float4(blendedPos, 1), viewProjMatrix);
	output.normal = normal;
	output.tex = input.tex;
//	output.color = input.blendIndices;

	return output;
}

float4 psSkinning(SkinningOutput input) : SV_TARGET
{
	return kdMap.Sample(linearSampler, input.tex);
//	return input.color.r + input.color.g + input.color.b + input.color.a;
	return input.color;
}

technique10 skinning
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsSkinningDqs() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psSkinning() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}
