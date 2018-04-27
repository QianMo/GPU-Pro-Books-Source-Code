#include "Platform.fxh"

Texture2D FontTex;

#ifdef MD_D3D10

SamplerState FontSamp
{
	Filter = MIN_MAG_LINEAR_MIP_POINT;
};

#elif defined( MD_D3D9 )

sampler2D FontSamp = sampler_state
{
	Texture = <FontTex>;
	MinFilter = LINEAR;
	MagFilter = LINEAR;
	MipFilter = NONE;
};

#endif

struct VSIn
{
	float2 pos	: POSITION;
	float2 texc	: TEXCOORD;
};

struct VSOut
{
	float4 pos 	: SV_Position;
	float2 texc	: TEXCOORD;
};

typedef VSOut PSIn;

VSOut main_vs( VSIn i )
{
	VSOut o;
	
	o.pos 	= float4( i.pos * 2, 0, 1); // to be able to get out of screen area
	o.texc 	= i.texc*0.5 + 0.5;

	return o;
}

float4 main_ps( PSIn i ) : SV_Target0
{
	return f4tex2D( FontTex, FontSamp, i.texc );
}

DepthStencilState DSS_Disable
{
	DepthEnable = FALSE;
};

DepthStencilState DSS_Default
{
};

BlendState BS_Enable
{
	BlendEnable[ 0 ] = TRUE;
	SrcBlend 		= SRC_ALPHA;
	DestBlend 		= INV_SRC_ALPHA;
	SrcBlendAlpha 	= SRC_ALPHA;
	DestBlendAlpha	= INV_SRC_ALPHA;
};

BlendState BS_Default
{
};

MD_TECHNIQUE main
{
	pass
	{
		SetVertexShader( CompileShader( MD_VS_TARGET, main_vs() ) );

#ifdef MD_D3D10
		SetGeometryShader( NULL );
#endif

		SetPixelShader( CompileShader( MD_PS_TARGET, main_ps() ) );

#ifdef MD_D3D10
		SetBlendState( BS_Enable, float4(0,0,0,0), 0xffffffff );
		SetDepthStencilState( DSS_Disable, 0 );
#elif defined( MD_D3D9 )
		ZEnable = FALSE;
		AlphaBlendEnable = TRUE;
		SrcBlend = SrcAlpha;
		DestBlend = InvSrcAlpha;
#endif
	}

	pass
	{

#ifdef MD_D3D10
		SetBlendState( BS_Default, float4(0,0,0,0), 0xffffffff );
		SetDepthStencilState( DSS_Default, 0 );
#else
		ZEnable = TRUE;
		AlphaBlendEnable = FALSE;
		SrcBlend 	= One;
		DestBlend 	= Zero;
#endif
	}
}