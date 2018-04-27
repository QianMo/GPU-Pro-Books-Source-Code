//
cbuffer perFrame
{
	float4x4 	matWVP;
	float4x4	matW;
	float3 		objLightDir;
	float3		objVPos;
	float3		worldVPos;
	float 		time;
	float		objFreq;
	float		objAmp;
};

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
				out float4 oPos 		: SV_Position,
				uniform float normScale,
				uniform float revNorm )
{
	for( int i = 0; i < 3; i ++ )
		oInvTBNW[i] = 0;

	pos.xyz += norm * normScale;

	pos.xyz += sin( pos.xyz * objFreq + time * 3.1415*2 * 10 ) * objAmp;

	oPos 	= mul( pos, matWVP );

	norm *= revNorm;

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

	float3 texc	= pos*0.25f+time*2;

	oInvTBNW[0][3][0] = texc.x;
	oInvTBNW[0][3][1] = texc.y;
	oInvTBNW[0][3][2] = texc.z;

#if DEBUG
	dl = light;
	dv = view;
#endif
	
}


Texture2D NormalTex;
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

	return normalize(val);
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

	n[0] = ReadNormal( iTexc.yz );
	n[1] = ReadNormal( iTexc.xz );
	n[2] = ReadNormal( iTexc.xy );

	float d = 0;
	float s = 0;

	float3 sceneColr = 0;

	float3 fact = normalize( abs(iFact) );
	fact *= fact;

#if DEBUG
	const int i = 0;
	v[i] = normalize( v[i] );
	l[i] = normalize( l[i] );

	dl = normalize( dl );
	dv = normalize( dv );

	d = saturate( dot( n[i], l[i] ) );
	s = pow( saturate( dot( reflect(-l[i], n[i]), v[i] ) ) * saturate( d * 1000. ), 32 );
	float sc = pow( saturate( dot( reflect(-dl, normalize(iFact) ), dv ) ) * saturate( d * 1000. ), 32 );
	return float4(	d, s*10, sc*10, 1);
#endif

	wview = normalize( wview );

	const int NUM = 3;

	[unroll]
	for( int i = 0; i < NUM; i++ )
	{
		v[i] = normalize( v[i] );
		l[i] = normalize( l[i] );

		float cd = saturate( dot( n[i], l[i] ) ) * fact[i];
		d += cd;
		s += pow( saturate( dot( reflect(-l[i], n[i]), v[i] ) ) * ( cd > 0 ), 64 ) * fact[i];

		float3 wn = mul( iInvTBNW[i], normalize(float3(n[i].xy,n[i].z*2)) );

		float3 refr = Refract( wview, wn, 0.075 );

		float3 refrColr = Scene.Sample( ss, refr );

		float3 refl = reflect( wview,  wn );

		float3 reflColr = Scene.Sample( ss, refl );

		float fres = dot( -wview, wn );

#if 0		
		fres *= fres;
#endif

		sceneColr += lerp( reflColr, refrColr, fres ) * fact[i];
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
#if 0
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs( 0.0, -1 ) ) );
		SetGeometryShader( NULL );
		SetPixelShader( MainPS );

		SetDepthStencilState( DSS_Disable, 0 );
		SetRasterizerState( RS_InvCulling );
	}
#endif

	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs( 0.1, 1 ) ) );
		SetGeometryShader( NULL );
		SetPixelShader( MainPS );

#if 1
		SetDepthStencilState( DSS_LessEqual, 0 );
		SetRasterizerState( RS_Default );
#else
		SetDepthStencilState( DSS_Disable, 0 );
		SetRasterizerState( RS_NoCulling );
#endif
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
		SetRasterizerState( RS_Default );
	}
}