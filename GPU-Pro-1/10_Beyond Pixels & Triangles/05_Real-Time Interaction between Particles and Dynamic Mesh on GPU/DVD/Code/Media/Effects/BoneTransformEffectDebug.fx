//

cbuffer perFrame
{
	float4x3 	matWVP;
	float4x3 	matArrBones[128];
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
	float4 pos 			: SV_Position;
	float3 norm			: NORMAL;
};

typedef VSOut PSIn;

VSOut main_vs(	const VSIn vi,
				uniform const uint numWeights )
{
	VSOut o;

	float4 pos = 0;
	float3 norm = 0;

#ifdef MD_POST_TRANSFORM
    pos = vi.pos;
	norm = vi.norm;
#else
	[unroll]
	for(uint i=0;i<numWeights;i++)
	{
		uint boneIdx = vi.indices[i];
		float weight = vi.weights[i];

		pos.xyz	+= mul(vi.pos, matArrBones[boneIdx]) * weight;
		norm 	+= mul(vi.norm, (float3x3)matArrBones[boneIdx]) * weight;
	}

	pos.w = 1;
#endif

	o.norm = norm;
	o.pos = mul( pos, matWVP );

	return o;
};


float4 main_ps( in PSIn i ) : SV_Target0
{
	return float4( 0, normalize(i.norm).y, 0, 1 );
}





technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs(4)) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );
	}

}