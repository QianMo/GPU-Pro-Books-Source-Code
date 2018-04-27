//

#define USE_INT 1

void main_vs( 	in float4 pos : POSITION,
				out float2 oTex : TEXCOORD,
				out float4 oPos : SV_Position )
{
	oTex = pos.xy*0.5+0.5;
	oPos = pos;
}

#if USE_INT
#define TTYPE <int>
#else
#define TTYPE
#endif

Texture3D TTYPE tex;


cbuffer Default
{
	float slice;
}

SamplerState ss
{
};


float4 main_ps( in float2 tc : TEXCOORD ) : SV_Target
{
#if USE_INT
	int3 sc = float3(tc.xy, slice)*128;
	int res = tex.Load( int4( sc, 0 ) );
	if(  res != -1) 
		return float4(res/65536., (res%256)/256., 0, 0 );
	else
		return 0;
#else
	return tex.SampleLevel( ss, float3( tc.xy, slice ), 0 ).a > 0.1;
#endif
}


DepthStencilState DSS_Disable
{
    DepthEnable 	= FALSE;
};

RasterizerState RS_CullDisabled
{
  	CullMode = None;
};

BlendState BS_Blend
{
	BlendEnable[0] = TRUE;
	SrcBlend 	= BLEND_FACTOR;
	DestBlend 	= INV_BLEND_FACTOR;

};




RasterizerState RS_Default
{
};

DepthStencilState DSS_Default
{
};

BlendState BS_Default
{
};


technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs()));
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, main_ps()));
		SetRasterizerState( RS_CullDisabled );	
		SetDepthStencilState( DSS_Disable, 0 );
		SetBlendState( BS_Blend, float4( 0.5f, 0.5f, 0.5f, 0.0f ), 0xFFFFFFFF );
	}
	
	pass
	{
		SetRasterizerState( RS_Default );
		SetDepthStencilState( DSS_Default, 0 );
		SetBlendState( BS_Default, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
	}
}