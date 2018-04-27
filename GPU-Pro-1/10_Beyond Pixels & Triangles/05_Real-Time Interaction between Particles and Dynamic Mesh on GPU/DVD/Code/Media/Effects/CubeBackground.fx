//
cbuffer def
{
#ifdef MD_RENDER_TO_CUBEMAP
	float4x4 matCubeMapInvVRotP[6];
#else
	float4x4 matInvVRotP;
#endif
}

void main_vs(	in float4 iPos 	: POSITION,
#ifdef MD_RENDER_TO_CUBEMAP
				in uint iInsID	: SV_InstanceID,
#endif
				out float3 oTex	: TEXCOORD,
				out float4 oPos	: SV_Position
#ifdef MD_RENDER_TO_CUBEMAP
				,
				out uint oInsID	: INSTANCE_ID
#endif
)
{

#ifdef MD_RENDER_TO_CUBEMAP
	float4x4 matInvVRotP = matCubeMapInvVRotP[iInsID];
#endif

	oTex 	= mul( float4(iPos.xy, 0, 1), matInvVRotP );
	oPos 	= float4(iPos.xy,1,1);
#ifdef MD_RENDER_TO_CUBEMAP
	oInsID	= iInsID;
#endif
}

#ifdef MD_RENDER_TO_CUBEMAP

struct GSIn
{
	float3 	texc	: TEXCOORD;
	float4 	pos		: SV_Position;
	uint	instID	: INSTANCE_ID;
};

struct GSOut
{
	float3 	texc	: TEXCOORD;
	float4 	pos		: SV_Position;
	uint 	rtID	: SV_RenderTargetArrayIndex;
};

[maxvertexcount(3)]
void main_gs( triangle GSIn ia[3], inout TriangleStream<GSOut> os )
{
	[unroll]
	for(int i = 0; i < 3; i ++ )
	{
		GSOut o;
		o.pos 		= ia[i].pos;
		o.texc  	= ia[i].texc;
		o.rtID		= ia[i].instID;
		os.Append( o );
	}
	os.RestartStrip();
}
#endif

SamplerState ss
{
	Filter = MIN_MAG_MIP_LINEAR;
};

TextureCube DiffuseTex;

float4 main_ps( in float3 iTex : TEXCOORD ) : SV_Target0
{
	return float4(DiffuseTex.Sample( ss, iTex ).rgb, 0 );
}

DepthStencilState DSS_LessEqual
{
	DepthFunc 		= LESS_EQUAL;
	DepthWriteMask 	= ZERO;
};

DepthStencilState DSS_Default
{
};



technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
#ifdef MD_RENDER_TO_CUBEMAP
		SetGeometryShader( CompileShader( gs_4_0, main_gs() ) );
#else
		SetGeometryShader( NULL );
#endif
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );

		SetDepthStencilState( DSS_LessEqual, 0 );
	}
	
	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
	
}