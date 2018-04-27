float3 rotByQuat( float3 v, float4 q )
{

	float3 uv, uuv;
	float3 qvec = q.xyz; 
	uv = cross( qvec, v );
	uuv = cross( qvec, uv );
	uv *= 2.0f * q.w;
	uuv *= 2.0f;

	return v + uv + uuv;
}

float4 quatFrom2Axes( float3 a1, float3 a2, float3 fallback )
{

	float4 res;

	float3 h = a1 + a2;

	float3 v = cross( a1, a2 );	

	[flatten]
	if( dot( v, v ) > 0.0001 )
	{
		[flatten]
		if( dot( h, h ) > 0.0001 )
		{
			float cos_a = dot( normalize(h), a1 );
			float sin_a = sqrt( 1 - cos_a*cos_a );

			res = float4( normalize(v.xyz) * sin_a, cos_a );
		}
		else
		{
			res = float4( fallback, 0 );
		}
	}
	else
	{
		res = float4( 0, 0, 0, 1 );
	}

	return res;
}