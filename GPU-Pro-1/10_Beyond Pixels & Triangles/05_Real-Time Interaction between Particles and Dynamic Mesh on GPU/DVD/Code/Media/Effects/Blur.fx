//

cbuffer def
{
	float2 step;
	float2 resolutionCorr;
}

struct VSIn
{
	float4 pos : POSITION;	
};

#define NUM_SAMPLES 3
#define NORMAL_NUM_SAMPLES 3

struct VSOut
{
	float4 pos 						: SV_Position;
	float4 texc_arr[NUM_SAMPLES]	: TEXCOORD;
	float2 texc 					: TEXCOORD_CENTRE;
};

struct VSOut_N
{
	float4 pos 							: SV_Position;
	float4 texc_arr[NORMAL_NUM_SAMPLES]	: TEXCOORD;
	float2 texc 						: TEXCOORD_CENTRE;
};


typedef VSOut PSIn;
typedef VSOut_N PSIn_N;

VSOut main_vs( VSIn i )
{
	VSOut o;
	o.pos = i.pos;
	// resolution corr brings us on texel boundaries, thus we use filtering as extra avaraging

	o.texc = float2( i.pos.x, -i.pos.y ) * 0.5 + resolutionCorr;

	[unroll]
	for( int ii = 0; ii < NUM_SAMPLES; ii++)
	{
		o.texc_arr[ii].xy = o.texc - step * (ii+1);
		o.texc_arr[ii].zw = o.texc + step * (ii+1);
	}

	return o;
}


VSOut_N normal_vs( VSIn i )
{
	VSOut_N o;
	o.pos = i.pos;
	// resolution corr brings us on texel boundaries, thus we use filtering as extra avaraging

	o.texc = float2( i.pos.x, -i.pos.y ) * 0.5 + 0.5;

	[unroll]
	for( int ii = 0; ii < NORMAL_NUM_SAMPLES; ii++)
	{
		o.texc_arr[ii].xy = o.texc - step * (ii+1) + resolutionCorr;
		o.texc_arr[ii].zw = o.texc + step * (ii+1) - resolutionCorr;
	}

	return o;
}


Texture2D tex;

SamplerState ss
{
	Filter = MIN_MAG_LINEAR_MIP_POINT;
	AddressU = CLAMP;
	AddressV = CLAMP;
	AddressW = CLAMP;
};

float4 main_ps( PSIn i ) : SV_Target
{
	float4 s = tex.SampleLevel( ss, i.texc, 0 );

	float coefs[NUM_SAMPLES] = { 0.8, 0.5, 0.3 };

	float sumw = 1;

	{
		for( int ii = 0; ii < NUM_SAMPLES; ii ++ )
			sumw += coefs[ii] * 2;
	}
	
	[unroll]
	for( int ii = 0; ii < NUM_SAMPLES; ii ++ )
	{
		s += tex.SampleLevel( ss, i.texc_arr[ii].xy, 0 ) * coefs[ii];
		s += tex.SampleLevel( ss, i.texc_arr[ii].zw, 0 ) * coefs[ii];
	}

	s /= sumw;

	return s;
}

float4 normal_ps( PSIn_N i ) : SV_Target
{

	float3 s;
	float orgw;

	float4( s, orgw ) = tex.SampleLevel( ss, i.texc, 0 );

	float coefs[ NORMAL_NUM_SAMPLES ] = { 0.8, 0.5, 0.3 };

	float sumw = 1;

#if 0
	{
		for( int ii = 0; ii < NORMAL_NUM_SAMPLES; ii ++ )
			sumw += coefs[ii] * 2;
	}
#endif
	
	[unroll]
	for( int ii = 0; ii < NORMAL_NUM_SAMPLES; ii ++ )
	{
		float4 v = tex.SampleLevel( ss, i.texc_arr[ii].xy, 0 );
		float dcoef = (1 - saturate( abs( v.w - orgw ) * 4 ) );
		sumw += coefs[ii] * dcoef;
		s += v * dcoef * coefs[ii];

		v = tex.SampleLevel( ss, i.texc_arr[ii].zw, 0 );
		dcoef = (1 - saturate( abs( v.w - orgw ) * 4 ) );
		sumw += coefs[ii] * dcoef;
		s += v * (1 - saturate( abs( v.w - orgw ) ) ) * coefs[ii];
	}

	s /= sumw;

	return float4(s, orgw);
}


DepthStencilState DSS_Disable
{
	DepthEnable = FALSE;
};

DepthStencilState DSS_Default
{
};

VertexShader MainVS = CompileShader( vs_4_0, main_vs() );
VertexShader NormalVS = CompileShader( vs_4_0, normal_vs() );
PixelShader MainPS = CompileShader( ps_4_0, main_ps() );
PixelShader NormalPS = CompileShader( ps_4_0, normal_ps() );

technique10 main
{
	pass
	{
		SetVertexShader( MainVS );
		SetGeometryShader( NULL );
		SetPixelShader( MainPS );

		SetDepthStencilState( DSS_Disable, 0 );
	}

	pass
	{
		SetVertexShader( MainVS );
		SetPixelShader( MainPS );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}

technique10 normal
{
	pass
	{
		SetVertexShader( NormalVS );
		SetGeometryShader( NULL );
		SetPixelShader( NormalPS );

		SetDepthStencilState( DSS_Disable, 0 );
	}

	pass
	{
		SetVertexShader( NormalVS );
		SetPixelShader( NormalPS );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}