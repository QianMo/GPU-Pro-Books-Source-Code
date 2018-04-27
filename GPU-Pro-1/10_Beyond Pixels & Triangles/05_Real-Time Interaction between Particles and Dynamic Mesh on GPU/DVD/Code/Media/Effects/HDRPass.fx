//

cbuffer def
{
	float2 step;
	float2 resolutionCorr;
	float rate;
}

struct VSIn
{
	uint id : SV_VertexID;
};

#define NUM_SAMPLES 4

struct VSOut
{
	float4 pos 						: SV_Position;
	float4 texc_arr[NUM_SAMPLES]	: TEXCOORD;
	float2 texc 					: TEXCOORD_CENTRE;
};

typedef VSOut PSIn;

VSOut main_vs( VSIn i )
{

	float4 val[4] = 
	{
		float4(-1,-1,0,1), float4(-1,+1,0,0),
		float4(+1,-1,1,1), float4(+1,+1,1,0)
	};


	VSOut o;
	o.pos = float4(val[ i.id ].xy, 0, 1 );

	o.texc = val[ i.id ].zw;

	// resolution corr brings us on texel boundaries, thus we use filtering as extra avaraging
	// ( remember pixel/texel coords match in D3D10 )
	// NOTE : we want centre pixel unfiltered to prevent bluring of non-bright pixels
	[unroll]
	for( int ii = 0; ii < NUM_SAMPLES; ii++)
	{
		o.texc_arr[ii].xy = o.texc - step * (ii+1)  + resolutionCorr;
		o.texc_arr[ii].zw = o.texc + step * (ii+1)  - resolutionCorr;
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

	float3 s;
	float alpha;

	float4( s, alpha ) = tex.SampleLevel( ss, i.texc, 0 );

#if 0
	s *= ( alpha * rate + 1 );
#endif

	float coefs[NUM_SAMPLES] = { 0.9, 0.6, 0.3, 0.2 };	
	
	[unroll]
	for( int ii = 0; ii < NUM_SAMPLES; ii ++ )
	{
		float4 v = tex.SampleLevel( ss, i.texc_arr[ii].xy, 0 );
		s += v * v.a * coefs[ii] * rate;

		v = tex.SampleLevel( ss, i.texc_arr[ii].zw, 0 );
		s += v * v.a * coefs[ii] * rate;
	}

	return float4( s, alpha );
}



DepthStencilState DSS_Disable
{
	DepthEnable = FALSE;
};

DepthStencilState DSS_Default
{
};

VertexShader MainVS = CompileShader( vs_4_0, main_vs() );
PixelShader MainPS = CompileShader( ps_4_0, main_ps() );

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

