//
cbuffer perFrame
{
	float4x4 	matWVP;
	float4x4	matW;
	float4x4	matVox;
	float4x4	matWVox;
	float3 		objLightDir;
	float3		objVPos;
	float3		worldVPos;
	float 		time;
	float		objFreq;
	float		objAmp;
	float3		voxStep;
	float3		toTex;
	float		refrK = 0.05;

};

float3 Normalize( float3 i )
{
#if 0
	if( dot(i, i) < 0.00001 )
		return float3(0,0,1);
	else
#endif
		return normalize( i );
}

float4 quatFrom2Axes( float3 a1, float3 a2, float3 fallback )
{

	float4 res;

	float3 h = a1 + a2;

	float3 v = cross( a1, a2 );	

	if( dot( v, v ) > 0.0001 )
	{
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

#define DEBUG 0

void main_vs( 	in float4 pos			: POSITION,
				in float3 norm			: NORMAL,
				out float4 oView[3]		: VIEWS_AND_WVIEW,
				out float4 oLight[3]	: LIGHTS_AND_FACT,
				out float4x3 oInvTBNW[3]: INV_TBNW_AND_TEXC,
#if DEBUG
				out float3 dl			: DL,
				out float3 dv			: DV,
#endif
				out float4 oPos 		: SV_Position )
{
	for( int i = 0; i < 3; i ++ )
		oInvTBNW[i] = 0;

	float4 voxPos = mul( pos, matWVox );

	oPos 	= mul( pos, matWVP );

	oLight[0].w = norm.x;
	oLight[1].w = norm.y;
	oLight[2].w = norm.z;

	float3 na[3] = { { 1, 0, 0 }, { 0, 1, 0 }, { +0, +0, +1 } };
	float3 ta[3] = { { 0, 0, 1 }, { 1, 0, 0 }, { -1, +0, +0 } };
	float3 ba[3] = { { 0, 1, 0 }, { 0, 0, 1 }, { +0, -1, +0 } };

	const float3 light 	= objLightDir;
	float3 view 		= objVPos - pos;

	float3 wview = mul( pos, matW ) - worldVPos;

	oView[0].w = wview.x;
	oView[1].w = wview.y;
	oView[2].w = wview.z;

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
		oView[i].xyz = mul( TBN, view );
		oLight[i].xyz = mul( TBN, light );

		// the order is reversed cause of ROW magor, column magor stuff
		(float3x3)oInvTBNW[i] = mul( transpose((float3x3)matW), transpose( TBN ) );
	}

	float3 texc	= pos*0.5+time*2;

	oInvTBNW[0][3][0] = texc.x;
	oInvTBNW[0][3][1] = texc.y;
	oInvTBNW[0][3][2] = texc.z;

	oInvTBNW[1][3][0] = voxPos.x;
	oInvTBNW[1][3][1] = voxPos.y;
	oInvTBNW[1][3][2] = voxPos.z;

#if DEBUG
	dl = light;
	dv = view;
#endif
	
}


Texture2D NormalTex;
Texture3D VoxTex;
TextureCube Scene;

SamplerState ss
{
	Filter = MIN_MAG_MIP_LINEAR;
};

#define SNORM 0

float3 ReadNormal( float2 texc )
{
	float3 val;

#if SNORM
	// texture is expected to hold _SNORM format
	val.xy = NormalTex.Sample( ss, texc );
	val.z = sqrt( 1 - dot( val.xy, val.xy ) );
#else
	val = ( NormalTex.Sample( ss, texc*0.5 ) - 0.5 );
	val += ( NormalTex.Sample( ss, texc*0.25 ) - 0.5 );
#endif

	return Normalize(val);
}

// simplified refraction routing( returns unnormalized v )
float3 Refract( float3 v, float3 n, float k )
{
	float dpres = dot( v, n );
	float3 pv =  dpres * n;
	float3 d = pv - v;

	[flatten]
	if( dpres < -0.0001 )
		return v + k * d;
	else       
		return v;
}

float3 GetRefrColr( float3 start, float3 dir )
{

	dir = Normalize( mul( dir, matVox ) ) * voxStep;

	start = start*toTex + 0.5;

	float3 texc = start;

	bool inside = true;
	bool hitShit = false;

	float count = 0;

	texc += dir*4;

#if 0
	float4 s = VoxTex.SampleLevel( ss, texc, 0 );

	if( s.a < 0.75 )
	{
		return float4( 1, 0, 0, 1 );
	}
	else
	{
		return float4( 0, 1, 0, 1 );
	}
#endif

	bool reflected = false;

	float fresnel = 1;
	float3 reflDir = float3(0,0,1);

	float stepsTraveled = 0;


	[loop]
	while( ! ( dot( texc < 0, 1 ) + dot( texc > 1, 1 ) ) )
	{
		float4 s = VoxTex.SampleLevel( ss, texc, 0 );
		stepsTraveled += 1;

		if( inside )
		{
			if( s.a < 0.75 )
			{
				float3 n = VoxTex.SampleLevel( ss, texc - float3( 0, 0, .5/64 ), 0 );				
				n = lerp( n*2-1, s*2-1, 0.5 );

				if( dot(n, n) < 0.00001 ) 
				{	
					hitShit = true;
					n = float3(0,0,1);
				}
				count += 1;

				dir = normalize( Refract( normalize(dir), -normalize( n ), -refrK ) ) * voxStep;
				inside = false;

				stepsTraveled = 0;
			}
		}
		else
		{
			if( s.a >= 0.75 )
			{

				float3 n = VoxTex.SampleLevel( ss, texc - float3( 0, 0, .5/64 ), 0 );
				n = lerp( n*2-1, s*2-1, 0.5 );

				if( dot(n, n) < 0.00001 )
				{
					hitShit = true;
					n = float3(0,0,1);

				}
				count += 1;

				n = normalize( n );
				dir = normalize( dir );

				// up to one extra reflection
				if( !reflected && stepsTraveled > 1  )
				{
					reflDir = mul( matVox, reflect( dir, n ) );
					fresnel = dot( -dir, n );

					reflected = true;
				}

				dir = normalize( Refract(dir, n, refrK ) ) * voxStep;
				inside = true;
				stepsTraveled = 0;
			}			
		}
		texc += dir;		
	}

	dir = mul( matVox, dir );

#if 0
	if( hitShit )
		return float4(0,1,0,0);
	else
		return float4(count > 3, 0, 0, 0 );
#endif

	return lerp( Scene.Sample( ss, reflDir ), Scene.Sample( ss, dir ), fresnel );

}

float4 main_ps( in float4 v_[3]				: VIEWS_AND_WVIEW,
				in float4 l_[3]				: LIGHTS_AND_FACT,
				in float4x3 iInvTBNW_[3]	: INV_TBNW_AND_TEXC
#if DEBUG
				,
				in float3 dl		: DL,
				in float3 dv		: DV 
#endif
					) : SV_Target

{
	float3 n[3];

	float3 v[3] = { v_[0].xyz,  v_[1].xyz, v_[2].xyz };
	float3 l[3] = { l_[0].xyz,  l_[1].xyz, l_[2].xyz };

	float3x3 iInvTBNW[3] = { (float3x3)iInvTBNW_[0], (float3x3)iInvTBNW_[1], (float3x3)iInvTBNW_[2] };

// extract stuff
	float3 wview = float3( v_[0].w, v_[1].w, v_[2].w );
	float3 iFact = float3( l_[0].w, l_[1].w, l_[2].w );
	float3 iTexc = float3( iInvTBNW_[0][3][0], iInvTBNW_[0][3][1], iInvTBNW_[0][3][2] );

	float3 voxPos = float3( iInvTBNW_[1][3][0], iInvTBNW_[1][3][1], iInvTBNW_[1][3][2] );

	n[0] = ReadNormal( iTexc.yz );
	n[1] = ReadNormal( iTexc.xz );
	n[2] = ReadNormal( iTexc.xy );

	float d = 0;
	float s = 0;

	float3 fact = Normalize( abs(iFact) );
	fact *= fact;

#if DEBUG
	const int i = 0;
	v[i] = Normalize( v[i] );
	l[i] = Normalize( l[i] );

	dl = Normalize( dl );
	dv = Normalize( dv );

	d = saturate( dot( n[i], l[i] ) );
	s = pow( saturate( dot( reflect(-l[i], n[i]), v[i] ) ) * saturate( d * 1000. ), 32 );
	float sc = pow( saturate( dot( reflect(-dl, Normalize(iFact) ), dv ) ) * saturate( d * 1000. ), 32 );
	return float4(	d, s*10, sc*10, 1);
#endif

	wview = Normalize( wview );

	const int NUM = 3;

	float3 wn = 0;

	[unroll]
	for( int i = 0; i < NUM; i++ )
	{
		v[i] = Normalize( v[i] );
		l[i] = Normalize( l[i] );

		wn += mul( iInvTBNW[i], n[i] ) * fact[i];

		float cd = saturate( dot( n[i], l[i] ) ) * fact[i];
		d += cd;
		s += pow( saturate( dot( reflect(-l[i], n[i]), v[i] ) ) * ( cd > 0 ), 64 ) * fact[i];

	}


	float3 sceneColr = 0;
	{
		wn = normalize( wn );

		float3 refr = Refract( wview, wn, refrK );

		float3 refrColr = GetRefrColr( voxPos, refr );

		float3 refl = reflect( wview,  wn );

		float3 reflColr = Scene.Sample( ss, refl );

		float fres = saturate( dot( -wview, wn ) );

		sceneColr = lerp( reflColr, refrColr, fres );
	}


	sceneColr *= 2;

	s *= 3;

	return float4( sceneColr + s, 0 );

#if 0
	float3 waterColr = lerp( float3(0.125,0.125,0.25), float3(0.4,0.6,1.0), d );

	return float4( lerp( sceneColr, sceneColr*0.5+waterColr, saturate(0.25 + s))+ s, 1 );
#endif

}

DepthStencilState DSS_Disable
{
	DepthEnable = FALSE;
};


DepthStencilState DSS_LessEqual
{
	DepthFunc = LESS_EQUAL;
};

DepthStencilState DSS_Default
{
};

RasterizerState RS_InvCulling
{
	CullMode = FRONT;
};

RasterizerState RS_NoCulling
{
	CullMode = NONE;
};


RasterizerState RS_Default
{
};

PixelShader MainPS = CompileShader( ps_4_0, main_ps() );


technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( NULL );
		SetPixelShader( MainPS );

		SetDepthStencilState( DSS_LessEqual, 0 );
		SetRasterizerState( RS_Default );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
		SetRasterizerState( RS_Default );
	}
}