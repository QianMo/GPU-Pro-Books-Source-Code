cbuffer def
{
	float 	width;
	float	halfPixelSize;
	float2	rcpDims;
	float2	pixelDisplace;

	float2	depthK;
};

Texture2D MotionTex;
Texture2D ScreenTex;
Texture2D DepthTex0;
Texture2D DepthTex1;

struct VSIn
{
	uint id : SV_VertexID;
};

struct VSOut
{
	float2 texc : TEXCOORD;
	float2 dir	: DIRECTION;
};

typedef VSOut GSIn;

struct GSOut
{
	float4 pos 	: SV_Position;
	float4 colr	: COLOR;
#ifdef MD_ACCOUNT_DEPTH
	float4 texc_depth : TEXC_DEPTH;
#endif
};

typedef GSOut PSIn;

SamplerState ss
{
	Filter = MIN_MAG_MIP_POINT;
	AddressU = CLAMP;
	AddressV = CLAMP;
};

VSOut main_vs( VSIn i )
{
	VSOut o;

	float2 texc = float2( fmod( i.id, width ), (i.id / width)-frac(i.id / width) ) * rcpDims + pixelDisplace;

	o.texc 	= texc;
	o.dir	= MotionTex.SampleLevel( ss, texc, 0 );

	return o;
}


float unpackDepth( float val )
{
	return depthK.y / ( val - depthK.x );
}

[maxvertexcount(4)]
void main_gs( in point GSIn ia[1], inout TriangleStream<GSOut> os )
{
	float l = length( ia[0].dir );
	if( l > 0 )
	{
		float2 ndir = normalize( ia[0].dir );
		float2 perp = float2( -ndir.y, ndir.x );

		GSOut oa[4];

		float2 pos = float2( ia[0].texc.x, 1 - ia[0].texc.y ) * 2 - 1;
		float2 epos = pos + ia[0].dir;

		oa[0].pos.xy = pos  - perp * halfPixelSize ;
		oa[1].pos.xy = pos  + perp * halfPixelSize ;
		oa[2].pos.xy = epos - perp * halfPixelSize ;
		oa[3].pos.xy = epos + perp * halfPixelSize ;

		float2 texc0 = ia[0].texc;
		float2 texc1 = float2( epos.x,-epos.y)*0.5 + 0.5;

		float4 colr = ScreenTex.SampleLevel( ss, texc0, 0 );

#ifdef MD_ACCOUNT_DEPTH

		float d0 = DepthTex0.SampleLevel( ss, texc0, 0 );
		float d1 = DepthTex1.SampleLevel( ss, texc1, 0 );

		float w0 = unpackDepth( d0 );
		float w1 = unpackDepth( d1 );

		oa[0].pos.zw = float2( d0, 1 );
		oa[1].pos.zw = float2( d0, 1 );
		oa[2].pos.zw = float2( d1, 1 );
		oa[3].pos.zw = float2( d1, 1 );

		oa[0].pos *= w0;
		oa[1].pos *= w0;
		oa[2].pos *= w1;
		oa[3].pos *= w1;

		oa[0].texc_depth.xy = texc0;
		oa[1].texc_depth.xy = texc0;
		oa[2].texc_depth.xy = texc1;
		oa[3].texc_depth.xy = texc1;

		oa[0].texc_depth.zw = oa[0].pos.zw - float2( 0.5, 0 );
		oa[1].texc_depth.zw = oa[1].pos.zw - float2( 0.5, 0 );
		oa[2].texc_depth.zw = oa[2].pos.zw - float2( 0.5, 0 );
		oa[3].texc_depth.zw = oa[3].pos.zw - float2( 0.5, 0 );
#endif

		[unroll]
		for( int i = 0; i < 4; i ++ )
		{

#ifndef MD_ACCOUNT_DEPTH
			oa[i].pos.zw = float2( 0, 1 );

#endif
			oa[i].colr = colr;
		}

		os.Append( oa[0] );
		os.Append( oa[1] );
		os.Append( oa[2] );
		os.Append( oa[3] );

		os.RestartStrip();
	}
}


float4 main_ps( PSIn i ) : SV_Target
{

#ifdef MD_ACCOUNT_DEPTH
	float d0 = DepthTex0.SampleLevel( ss, i.texc_depth.xy, 0 );
	float d1 = i.texc_depth.z / i.texc_depth.w;

	clip( d0 - d1 );
#endif

	return float4( i.colr.rgb, 1 );
}

DepthStencilState DSS_Disabled
{
	DepthEnable = FALSE;
};

DepthStencilState DSS_Default
{
};

BlendState BS_Blend
{
	BlendEnable[0] 	= TRUE;
	SrcBlend 		= ONE;
	DestBlend 		= ONE;
	SrcBlendAlpha 	= ONE;
	DestBlendAlpha 	= ONE;
	
};

BlendState BS_Default
{
};


technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( CompileShader( gs_4_0, main_gs() ) );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );

		SetDepthStencilState( DSS_Disabled, 0 );
		SetBlendState( BS_Blend, float4(1,1,1,1), 0xffffffff );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
		SetBlendState( BS_Default, float4(1,1,1,1), 0xffffffff );
	}
}