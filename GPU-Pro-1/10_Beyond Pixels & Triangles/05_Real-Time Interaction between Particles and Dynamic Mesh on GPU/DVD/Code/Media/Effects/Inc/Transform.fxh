

void BoneTransformWithBoneArray( inout float4 pos, inout float3 norm, inout float3 tang, in uint4 indices, in float4 weights, uint numWeights, float4x3 boneArray[128] )
{
	float4 rpos 	= 0;
	float3 rnorm 	= 0;
	float3 rtang	= 0;

	[unroll]
	for( uint i = 0; i < numWeights; i ++ )
	{
		float4x3 mat = boneArray[ indices[i] ];

		float w = weights[ i ];

		rpos.xyz 	+= mul( pos, mat ) * w;
		rnorm		+= mul( norm, (float3x3)mat ) * w;
		rtang		+= mul( tang, (float3x3)mat ) * w;
	}

	rpos.w 	= 1;

	pos 	= rpos;
	norm 	= normalize( rnorm );
	tang 	= normalize( rtang );
}


void BoneTransform( inout float4 pos, inout float3 norm, inout float3 tang, in uint4 indices, in float4 weights, uint numWeights )
{
	BoneTransformWithBoneArray( pos, norm, tang, indices, weights, numWeights, matArrBones );
}

uint2 PackNormFloat3ToUInt2( float3 val )
{

	// DX9 documented signed packing. -1 -> -32767, not -32768
	int3 ival = val * 32767;

	uint2 result = 0;

	result.x = ( ival.x & 0xffff ) + ( ival.y << 16 );
	result.y = ival.z & 0xffff;

	return result; 
}

