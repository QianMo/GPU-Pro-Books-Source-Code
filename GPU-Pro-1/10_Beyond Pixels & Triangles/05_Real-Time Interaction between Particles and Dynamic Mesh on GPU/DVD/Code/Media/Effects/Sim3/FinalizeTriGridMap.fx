#include "TriGrid.fxh"

struct VSOut 
{
	float4 pos : SV_Position;
};

typedef VSOut PSIn;

VSOut main_vs()
{
	VSOut o;
	o.pos = float4( displacement, 0, 0, 1 );

	return o;
}

float main_ps( PSIn i ) : SV_Target
{
#ifdef MD_R16_UNORM_BLENDABLE
	return 1./65535;
#else
	return 1.;
#endif
}

BlendState BS_Add
{
    BlendEnable[0] = TRUE;
	SrcBlend = ONE;
    DestBlend = ONE;
    SrcBlendAlpha = ONE;
    DestBlendAlpha = ONE;
};


BlendState BS_Default
{
};

DepthStencilState DSS_Disabled
{
	DepthEnable = FALSE;
};


DepthStencilState DSS_Default
{
	
};

technique10 main
{
 	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );

		SetBlendState( BS_Add, float4( 0, 0, 0, 0 ), 0xFFFFFFFF );
		SetDepthStencilState( DSS_Disabled, 0 );
	}

	pass
	{
		SetBlendState( BS_Default, float4( 0, 0, 0, 0 ), 0xFFFFFFFF );
		SetDepthStencilState( DSS_Default, 0 );
	}
}