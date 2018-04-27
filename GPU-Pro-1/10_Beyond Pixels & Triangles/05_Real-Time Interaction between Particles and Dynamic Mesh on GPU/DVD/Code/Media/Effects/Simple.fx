//
cbuffer perFrame
{
	float4x4 matWVP;
	float4x3 matWV;
	float4 color = float4(0.8,0.8,0,0);
};

void main_vs( 	in float4 pos		: POSITION,
				in float3 norm		: NORMAL,
				out float3 oNorm	: NORMAL,
				out float3 oView	: VIEW,
				out float4 oPos 	: SV_Position )
{
	oPos 	= mul( pos, matWVP );
	oView	= -mul( pos, matWV );
	oNorm 	= mul( norm, matWV );
}

float4 main_ps( 
				in float3 norm : NORMAL,
				in float3 view : VIEW  ) : SV_Target
{
	norm = normalize(norm);
	view = normalize(view);

	float3 light = float3(0,0,-1);
	
	float d = saturate(dot( norm, light ));

	float3 reflLight = reflect( -light, norm );

	float s = saturate( dot( reflLight, view ) );

	return d * color + pow(s, 16)*0.7;
}


DepthStencilState DSS_LessEqual
{
	DepthFunc = LESS_EQUAL;
};

DepthStencilState DSS_Default
{
};


//------------------------------------------

technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );

		SetDepthStencilState( DSS_LessEqual, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}