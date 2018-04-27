#include "TriGrid.fxh"

struct VSIn
{
	uint idx : INDEX;
};

struct VSOut 
{
	float4 pos : SV_Position;
};


VSOut main_vs( VSIn i )
{
	VSOut o;

	uint index = i.idx;

	float posx;

	index = index >> mask;

	posx = index & 1 ? 3. : index * rcpNumCells + displacement;

	o.pos = float4( posx-1.0, 0, 0, 1 );

	return o;
}

float main_ps() : SV_Target
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