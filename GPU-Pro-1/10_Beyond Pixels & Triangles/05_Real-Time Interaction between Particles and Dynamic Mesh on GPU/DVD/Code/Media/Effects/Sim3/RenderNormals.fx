//

cbuffer def
{
	float2 size; // y contains size squared

	float4x3 matWV;
	float4x3 invMatV;
	float4x4 matP;
}

struct VSIn
{
	float3 pos		: POSITION;
	float3 speed	: SPEED;
};

struct VSOut
{
	float3 	pos0	: POSITION0;
	float3	pos1	: POSITION1;
};

typedef VSOut GSIn;

struct GSOut
{
	float4 pos 			: SV_Position;
	float2 texc			: TEXCOORD;
	float  w			: W;
	float3 transform1	: TRANSFORM1;
	float3 transform2 	: TRANSFORM2;
};

typedef GSOut PSIn;

VSOut main_vs( VSIn i )
{
	VSOut o;

	o.pos0		= mul( float4( i.pos.xyz, 1 ), matWV );
	o.pos1		= mul( float4( i.pos.xyz + i.speed*dot( i.speed, i.speed ) * 0.0015625, 1 ), matWV );
	
	return o;
}

[maxvertexcount(4)]
void main_gs( in point GSIn ia[1], inout TriangleStream<GSOut> os )
{

	float2 disp = ia[0].pos1 - ia[0].pos0;

	float3 poses[4] = { ia[0].pos0, ia[0].pos1, ia[0].pos1, ia[0].pos1 };

	float2 ndir = float2( 0, 1 );
	float2 perp = float2( 1, 0 );

	float l = length( disp );

	[flatten]
	if( l > 0.0000001 )
	{
		ndir = disp / l;
		perp = float2( -ndir.y, ndir.x );
	}

	poses[0].xy += ( - 0 	- ndir ) * size.x ;
	poses[1].xy += ( - perp + 0    ) * size.x ;
	poses[2].xy += ( + perp + 0    ) * size.x ;
	poses[3].xy += ( + 0    + ndir ) * size.x ;

	float2 texces[4] = { { 0, 1 }, { 1, 1 }, { 0, 0 }, { 1, 0 } };

	float3x3 toTBN;

	toTBN._11_21_31 = normalize( poses[3] - poses[0] );
	toTBN._12_22_32 = -float3( perp, 0 );
	toTBN._13_23_33 = cross( toTBN._11_21_31, toTBN._12_22_32 );

	[unroll]
	for( int i = 0; i < 4; i ++ )
	{
		GSOut o;
		o.pos 			= mul( float4( poses[ i ], 1 ), matP );
		o.w				= o.pos.w ; // reading SV_Position in PS was slow some time ago
		o.texc			= texces[i];

		o.transform1	= toTBN._11_12_13;
		o.transform2	= toTBN._21_22_23;

		os.Append( o );
	}

	os.RestartStrip();
}

SamplerState ss
{
	Filter = MIN_MAG_MIP_LINEAR;
};

Texture2D NormTex;

float4 main_ps( PSIn i ) : SV_Target0
{
	float4 val = NormTex.Sample( ss, i.texc );

	clip( val.r - 0.4 );

	float2 norm_xy = ( val.ga - 0.5 ) * 2;

	float3 norm = float3( norm_xy, sqrt( saturate( 1 - dot( norm_xy, norm_xy ) ) ) );

	norm = normalize( float3(norm.xy, norm.z*0.66 ) );
	
	float3 vnorm;

	vnorm.x = dot( i.transform1, norm );
	vnorm.y	= dot( i.transform2, norm );

	// fight for less imports: no transform3, cause in our case .z always stares at the viewer
	vnorm.z = -sqrt( saturate( 1 - dot( vnorm.xy, vnorm.xy ) ) );

	return float4( vnorm, i.w );
}

RasterizerState RS_NoCulling
{
	CullMode = NONE;
};

RasterizerState RS_Default
{
	
};

technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( CompileShader( gs_4_0, main_gs() ) );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );

		SetRasterizerState( RS_NoCulling );
	}

	pass
	{
		SetRasterizerState( RS_Default );
	}
}