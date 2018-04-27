// 

// constant buffers


cbuffer perFrame
{
	float4x4 	matWVP;
	float3		objLightDir0= float3(0.7,0,-0.7);
	float3		objCamPos;
	float		specPower 	= 32;
	float		ambient		= 0.35;
	float3		specColr	= float3(.7,.7,.7);
};

cbuffer DieLetterz
{
	float4		letterPoses[512];
};


// input/output structs

struct VSIn
{
	uint bufIdx : VERTEX_IDX;
	uint letIdx : SYMBOL_IDX;
};

struct VSOut
{
	float4 pos 		: SV_Position;
	float3 tlight 	: LIGHT;
	float3 thalf	: HALF;
	float2 texc		: TEXCOORD;
};

// textures & buffer

Buffer<float4> FontBuf;
Texture2D NormalTex;
Texture2D DiffuseTex;

SamplerState defSampler
{
	Filter 			= ANISOTROPIC;
	MaxAnisotropy 	= 8;
};

typedef VSOut PSIn;

VSOut main_vs( const VSIn vi )
{
	VSOut vo;
	uint idx = vi.bufIdx;

	float4 pos 		= FontBuf.Load( idx++ );
	float3 norm 	= FontBuf.Load( idx++ ).xyz;
	float3 tang		= FontBuf.Load( idx++ ).xyz;
	float2 texc		= FontBuf.Load( idx ).xy;

	float3 binorm 	= cross(tang, norm);

	float3x3 matTBN = float3x3( tang, binorm, norm );

	pos.xyz *= 	letterPoses[vi.letIdx].z;
	pos.xy 	+= 	letterPoses[vi.letIdx].xy;

	vo.pos 		= mul( pos, matWVP );
	vo.tlight 	= normalize(mul( matTBN, objLightDir0 ));
	vo.thalf	= normalize(mul( matTBN, normalize( objLightDir0 + (objCamPos - pos.xyz))));
	vo.texc		= texc;

	return vo;	
}

float4 main_ps ( const PSIn pi ) : SV_Target0
{
	float3 norm = NormalTex.Sample( defSampler, pi.texc ).xyz*(2).xxx-(1).xxx;
	float diff 	= saturate(dot(norm, pi.tlight));
	float spec 	= pow(saturate(dot(norm, normalize(pi.thalf))), specPower);
	return 4*(DiffuseTex.Sample(defSampler, pi.texc)*(diff+ambient) + float4(specColr,0)*spec);
}

RasterizerState MSEnabled
{
	MultiSampleEnable = TRUE;
};

RasterizerState Default
{
};


technique10 main
{
	pass
	{
		SetVertexShader(CompileShader(vs_4_0, main_vs()));
		SetGeometryShader(NULL);
		SetPixelShader(CompileShader(ps_4_0, main_ps()));

		SetRasterizerState(MSEnabled);
	}

	pass
	{
		SetRasterizerState(Default);
	}
}