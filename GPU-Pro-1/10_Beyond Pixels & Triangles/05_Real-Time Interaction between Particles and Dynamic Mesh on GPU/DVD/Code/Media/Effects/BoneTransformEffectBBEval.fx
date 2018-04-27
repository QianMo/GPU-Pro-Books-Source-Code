//

cbuffer perFrame
{
	float4x3 	matW;

	float4x3 	matArrBones[128];

	float 		normScale;
	float 		time;
	float		objFreq;
	float		objAmp;
}


struct VSIn
{
	float4 pos 		: POSITION;
	float3 norm 	: NORMAL;
	uint4  indices	: BLENDINDICES;
	float4 weights	: BLENDWEIGHT;
};

struct VSOut
{
	float3 pos 			: POSITION;
	float3 norm			: NORMAL;
	float3 boned_pos    : BONED_POS;
	float4 sv_pos   	: SV_Position;
};

typedef VSOut PSIn;

VSOut main_vs(	const VSIn vi,
				uniform const uint numWeights )
{
	VSOut o;

	// account bone transforms

	float4 pos 		= 0;
	float3 norm		= 0;

	[unroll]
	for(uint i=0;i<numWeights;i++)
	{
		uint boneIdx = vi.indices[i];
		float weight = vi.weights[i];

		pos.xyz	+= mul(vi.pos, matArrBones[boneIdx]) * weight;
		norm 	+= mul(vi.norm, (float3x3)matArrBones[boneIdx]) * weight;
	}

	pos.w = 1;

	norm = normalize(norm);

	pos.xyz += norm.xyz * normScale;
	pos.xyz += sin( pos.xyz * objFreq + time  ) * objAmp;

	o.boned_pos = pos;

	o.pos = mul( pos, matW );
	o.norm = mul( norm, matW );
	o.sv_pos = float4( 0.5, 0.5, 0, 1 );

	return o;
};

struct PSOut
{
	float3 minExt : SV_Target0;
	float3 maxExt : SV_Target1;
};

PSOut main_ps( in PSIn i )
{
	PSOut o;
	o.minExt = -i.boned_pos;
	o.maxExt = i.boned_pos;
	return o;
};



DepthStencilState DSS_Disable
{
	DepthEnable = FALSE;
};


DepthStencilState DSS_Default
{
};

BlendState BS_MinMax
{
	BlendEnable[0] = TRUE;
	BlendEnable[1] = TRUE;

	SrcBlend = ONE;
	DestBlend = ONE;
	
	BlendOp = MAX;
};

BlendState BS_Default
{
};


technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs(4)) );
		SetGeometryShader( ConstructGSWithSO( CompileShader(vs_4_0, main_vs(4)), "POSITION.xyz;NORMAL.xyz") );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );

		SetDepthStencilState( DSS_Disable, 0 );
		SetBlendState( BS_MinMax, float4(0,0,0,0), 0xFFFFFFFF );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
		SetBlendState( BS_Default, float4(0,0,0,0), 0xFFFFFFFF );
	}
}