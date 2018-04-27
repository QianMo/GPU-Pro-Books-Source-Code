//

cbuffer def
{
	float dt;
}

struct VSIn
{
	float4 pos : POSITION;	
};

struct VSOut
{
	float4 pos : SV_Position;
	float2 tex : TEXCOORD;
};

typedef VSOut PSIn;

VSOut main_vs( VSIn i )
{
	VSOut o;
	o.pos = i.pos;
	o.tex = float2( i.pos.x, -i.pos.y ) * 0.5 + 0.5;

	return o;
}

Texture2D Prev;
Texture2D Curr;

SamplerState ss
{
	Filter = MIN_MAG_LINEAR_MIP_POINT;
	AddressU = CLAMP;
	AddressV = CLAMP;
	AddressW = CLAMP;
};

float4 main_ps( PSIn i ) : SV_Target
{
	float4 p = Prev.SampleLevel( ss, i.tex, 0 );
	float4 c = Curr.SampleLevel( ss, i.tex, 0 );

	return lerp( p, c, saturate( dt*24 ) );
}

DepthStencilState DSS_Disable
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

		SetDepthStencilState( DSS_Disable, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}

