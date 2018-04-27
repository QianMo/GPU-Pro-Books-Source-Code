//

#include "Math.fxh"

Texture2D NormalTex;

cbuffer Default
{
	float4x3 matT;
	float4x4 matWVP;
	float shift;
	float time;
};

struct VSIn
{
	float4 pos 	: POSITION;
	float3 norm	: NORMAL;
};

struct VSOut
{
	float4 		pos			: SV_Position;
	float3		texc		: TEXCOORD;
	float  		posw 		: POSITION;
	float3x2 	NT[3]		: NT; // normal transform
	float3		norm		: NORM;
	
};

typedef VSOut PSIn;

VSOut main_vs( in VSIn i )
{

	VSOut o;

	o.pos	= mul( i.pos, matWVP );
	o.posw 	= o.pos.w + shift;
	o.norm	= i.norm;
	o.texc = i.pos*2 + time;

	float3 na[3] = { { 1, 0, 0 }, { 0, 1, 0 }, { +0, +0, +1 } };
	float3 ta[3] = { { 0, 0, 1 }, { 1, 0, 0 }, { -1, +0, +0 } };
	float3 ba[3] = { { 0, 1, 0 }, { 0, 0, 1 }, { +0, -1, +0 } };

	float3 norm = i.norm;

	[unroll]
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

		b = cross( n, t );

		float3x3 TBN = { t, b, n };

		// the order is reversed cause of ROW magor, column magor stuff
		o.NT[i] = mul(transpose(matT),transpose(TBN));
	}

	return o;
}

SamplerState ss
{
	Filter = MIN_MAG_MIP_LINEAR;
};

float3 ReadNormal( float2 texc )
{
	float3 val;

	val = ( NormalTex.Sample( ss, texc ) - 0.5 );
	val += ( NormalTex.Sample( ss, texc*2 ) - 0.5 );

	val.z = sqrt( saturate( 1 - dot( val.xy, val.xy ) ) ) * 0.25;

	return normalize( val );
}


float4 main_ps( PSIn i ) : SV_Target
{

	float3 n[3];

	n[0] = ReadNormal( i.texc.yz );
	n[1] = ReadNormal( i.texc.xz );
	n[2] = ReadNormal( i.texc.xy );

	float3 normal = 0;

	float3 weights = abs(normalize(i.norm));

	[unroll]
	for( int ii = 0; ii < 3; ii ++ )
	{
		float3x3 NT;
		NT._11_21_31 = normalize(i.NT[ii]._11_21_31);
		NT._12_22_32 = normalize(i.NT[ii]._12_22_32);
		NT._13_23_33 = cross( i.NT[ii]._11_21_31, i.NT[ii]._12_22_32 );

		normal += mul( NT, n[ii] ) * weights[ii];
	}

	normal = normalize( normal );

	// not really necessery, but if we accumulate and put in some weird mesh
	// nans can spread and cause troubles
	[flatten]
	if( isnan( normal.x ) ) normal = float3(0,1,0);


	return float4( normal, i.posw );
} 

DepthStencilState DSS_LessEqual
{
	DepthFunc = LESS_EQUAL;
};

DepthStencilState DSS_Default
{
};


technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, main_ps()) );

		SetDepthStencilState( DSS_LessEqual, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}