#include "Velocity.fxh"

#if defined(MD_VTEXT)

cbuffer DieLetterz
{
	float4		letterPoses[512];
};

Buffer<float4> FontBuf;

struct VSIn
{
	uint bufIdx : VERTEX_IDX;
	uint letIdx : SYMBOL_IDX;
};

#elif defined(MD_BONE_TRANSFORM)

cbuffer BoneBuffer
{
	float4x3 matArrBones[128];
	float4x3 matArrPrevBones[128];
};

#include "Transform.fxh"

struct VSIn
{
	float4 pos 		: POSITION;
#ifdef MD_BONE_POST_TRANSFORM
	float4 traPos	: POSITION1;
#endif
	uint4 indices	: BLENDINDICES;
	float4 weights	: BLENDWEIGHT;
};

#else

struct VSIn
{
	float4 pos : POSITION;
};

#endif


struct VSOut
{
	float4 pos 		: SV_Position;
	float4 pos_c	: POS_CURR;
	float4 pos_p 	: POS_PREV; // sometime long ago it appeared that reading custom interpolator is faster than SV_Position
};

typedef VSOut PSIn;

VSOut main_vs( VSIn i )
{
	VSOut o;

	float4 pos, prevPos;

#if defined(MD_VTEXT)
	uint idx = i.bufIdx;

	pos 			= FontBuf.Load( idx++ );
	float3 norm 	= FontBuf.Load( idx++ ).xyz;
	float3 tang		= FontBuf.Load( idx++ ).xyz;
	float2 texc		= FontBuf.Load( idx ).xy;

	pos.xyz *= 	letterPoses[i.letIdx].z;
	pos.xy 	+= 	letterPoses[i.letIdx].xy;

	prevPos = pos;
#elif defined(MD_BONE_TRANSFORM)
	float3 dummyN = 0, dummyT = 0;
	pos = i.pos;
	prevPos = pos;

// current position is already skinned
#ifdef MD_BONE_POST_TRANSFORM
	pos = i.traPos;
#else
	BoneTransformWithBoneArray( pos, dummyN, dummyT, i.indices, i.weights, 4, matArrBones 			);
#endif

	BoneTransformWithBoneArray( prevPos, dummyN, dummyT, i.indices, i.weights, 4, matArrPrevBones 	);
#else
	pos = i.pos;
	prevPos = pos;
#endif

	o.pos = mul( pos, matWVP );

	o.pos_c = o.pos;
	o.pos_p = mul( prevPos, matPrevWVP );


	return o;
}

float2 main_ps( PSIn i ) : SV_Target
{
	float2 diff = i.pos_p.xy / i.pos_p.w - i.pos_c.xy / i.pos_c.w;
#if 0
	return float2( diff.x, -diff.y ) * 0.5;
#else
	return diff;
#endif
}

technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );
	}
}