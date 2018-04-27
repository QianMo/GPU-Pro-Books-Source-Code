float3 GetDNorm( float3 pos, float3 norm )
{
#if 0
	norm = normalize( norm );

	float3 fact = abs(norm);

	pos *= float3( 1, 1, 1 ) * 4;

	float2 texc[3] = { pos.xy, pos.zx, pos.yz };

	float3 na[3] = { { 1, 0, 0 }, { 0, 1, 0 }, { +0, +0, +1 } };
	float3 ta[3] = { { 0, 0, 1 }, { 1, 0, 0 }, { -1, +0, +0 } };
	float3 ba[3] = { { 0, 1, 0 }, { 0, 0, 1 }, { +0, -1, +0 } };

	float3 dn = 0;

	for( int i = 0; i < 3; i ++ )
	{
		float3 n = na[i];
		float3 t = ta[i];
		float3 b = ba[i];

		if( dot( n, norm ) < 0 )
		{
			n = -n;
			t = -t;
		}
			
		
		float4 q = quatFrom2Axes( n, norm, b );
		n = norm;
		t = rotByQuat( t, q );
		b = rotByQuat( b, q );

		float3x3 TBN = { t, b, n };

		float f = fact[i];

		float3 tnorm = Norm.Sample( ssw, texc[i] + time ) * 2 - 1;
		tnorm += Norm.Sample( ssw, texc[i] *.5 - time ) * 2 - 1;

		tnorm *= f*f;

		// get that to world
		dn += mul( tnorm, TBN );
	}

	return normalize( dn );
#else
	return 0;
#endif
	
}
