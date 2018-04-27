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
	float3 pos 	: POSITION;
	float3 norm	: NORMAL;
};

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

	pos.xyz *= 0.05;

	o.pos = mul( pos, matW );
	o.norm = mul( norm, matW );

	return o;
};

DepthStencilState DSS_Disable
{
	DepthEnable = FALSE;
};


DepthStencilState DSS_Default
{
};


technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs(4)) );
		SetGeometryShader( ConstructGSWithSO( CompileShader(vs_4_0, main_vs(4)), "POSITION.xyz;NORMAL.xyz") );
		SetPixelShader(NULL);

		SetDepthStencilState( DSS_Disable, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}