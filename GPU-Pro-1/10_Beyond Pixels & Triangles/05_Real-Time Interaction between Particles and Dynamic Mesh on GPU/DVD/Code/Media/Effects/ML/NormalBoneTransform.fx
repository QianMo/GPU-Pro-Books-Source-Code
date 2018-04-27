#include "ML.fxh"
#include "Transform.fxh"

struct VSIn
{
	float4 pos 		: POSITION;
	float3 norm		: NORMAL;
	float3 tang 	: TANGENT;

	uint4 indices	: BLENDINDICES;
	float4 weights	: BLENDWEIGHT;
};

struct VSOut
{
	float3 pos 		: POSITION;

	// normal and tangent get packed into 16 bits snorm
	uint2 norm		: NORMAL; 
	uint2 tang 		: TANGENT;
};

VSOut main_vs( VSIn i )
{
	float4 position = i.pos;
	float3 tangent 	= i.tang;
	float3 normal 	= i.norm;

	BoneTransform( position, tangent, normal, i.indices, i.weights, MD_NUM_WEIGHTS );

	VSOut o;

	o.pos		= position;

	o.norm 		= PackNormFloat3ToUInt2( normal );
	o.tang 		= PackNormFloat3ToUInt2( tangent );

	return o;
}

DepthStencilState DSS_Disable
{
	DepthEnable = FALSE;
};

DepthStencilState DSS_Default
{
};

VertexShader MainVS = CompileShader( vs_4_0, main_vs() );

technique10 main
{
	pass
	{
		SetVertexShader( MainVS );
		SetGeometryShader( ConstructGSWithSO( MainVS, "POSITION.xyz;NORMAL.xy;TANGENT.xy" ) );
		SetPixelShader( NULL );

		SetDepthStencilState( DSS_Disable, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}