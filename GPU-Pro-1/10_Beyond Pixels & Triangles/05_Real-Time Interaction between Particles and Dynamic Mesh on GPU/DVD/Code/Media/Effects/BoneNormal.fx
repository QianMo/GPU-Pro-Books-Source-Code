// textures

Texture2D NormalTex;
Texture2D DiffuseTex;


SamplerState defSampler
{
	Filter 			= ANISOTROPIC;
	MaxAnisotropy 	= 8;
};

cbuffer perFrame
{

	float4x4 	matWVP;
	float4x4	matVP;
	float3		objLightDir;		// Light in object space
	float3 		objCamPos;
	float		specPower 	= 15;
	float3		specColr	= float3(2,0.5,0.5);


	float4x3 	matArrBones[128];
}


struct VSInput
{
	float4 pos 		: POSITION;
	float3 norm 	: NORMAL;
	float3 tang 	: TANGENT;
	float2 texc		: TEXCOORD;
	uint4  indices	: BLENDINDICES;
	float4 weights	: BLENDWEIGHT;
};


struct VSOutput
{
	float4 pos 		: SV_Position;
	float3 tlight	: LIGHT;		// tangent space light
	float3 thalf	: HALF;			// tangent space half-vec
	float2 texc		: TEXCOORD;
};

typedef VSOutput PSInput;

VSOutput main_vs(	const VSInput vi,
					uniform const uint numWeights )
{
	VSOutput o;

	// account bone transforms

	float4 pos 		= 0;
	float3 norm		= 0;
	float3 tang		= 0;

	[unroll]
	for(uint i=0;i<numWeights;i++)
	{
		uint boneIdx = vi.indices[i];
		float weight = vi.weights[i];

		pos.xyz	+= mul(vi.pos, matArrBones[boneIdx]) * weight;
		norm 	+= mul(vi.norm, (float3x3)matArrBones[boneIdx]) * weight;
		tang	+= mul(vi.tang, (float3x3)matArrBones[boneIdx]) * weight;
	}

	pos.w = 1;

	norm = normalize(norm);
	tang = normalize(tang);

	float3 binorm = cross(norm, tang);

	// do the lighting
	
	o.pos 	= mul(pos, matWVP);
	o.texc  = vi.texc;

	float3x3 toTBN 	= float3x3( tang, binorm, norm );
	o.tlight		= normalize(mul(toTBN, objLightDir)); 	// note : we dont minus, this is done by param provider
	o.thalf			= normalize(mul(toTBN, objLightDir+normalize(objCamPos-pos)));

	return o;
};

float4 main_ps( PSInput pi) : SV_Target0
{
	float3 norm = (NormalTex.Sample(defSampler, pi.texc).xyz*(2).xxx-(1).xxx);
	float lit	= saturate(dot(pi.tlight, norm));
	float spec	= pow(saturate(dot(normalize(pi.thalf), norm)), specPower);
	return (lit.xxxx * DiffuseTex.Sample(defSampler, pi.texc)+float4(specColr*spec.xxx,0));
}

RasterizerState RS_MSAAEnabled
{
	MultiSampleEnable = TRUE;
};

RasterizerState RS_Default
{
};


technique10 main
{
	pass
	{
		SetVertexShader(CompileShader( vs_4_0, main_vs(4)));
		SetGeometryShader(NULL);
		SetPixelShader(CompileShader(ps_4_0, main_ps()));

		SetRasterizerState(RS_MSAAEnabled);
	}

	pass
	{
		SetRasterizerState(RS_Default);
	}
}