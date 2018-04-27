//
cbuffer def
{
	int clearVal;
}


struct VSOut
{
	float4 pos 	: SV_Position;
	uint iid	: IID;
};


VSOut main_vs(	float2 pos : POSITION,
				uint iidx : SV_InstanceID )
{
	VSOut o;
	o.pos = float4( pos, 0, 1 );
	o.iid = iidx;

	return o;
}

typedef VSOut GSIn;


struct GSOut
{
	float4 pos 	: SV_Position;
	uint ridx	: SV_RenderTargetArrayIndex;
};

[maxvertexcount(3)]
void main_gs( triangle GSIn ia[3], inout TriangleStream<GSOut> os )
{
	GSOut o;
	o.ridx = ia[0].iid;

	[unroll]
	for( int i = 0; i < 3; i++ )
	{
		o.pos = ia[i].pos;
		os.Append( o );
	}
	os.RestartStrip();
}

int main_ps() : SV_Target
{
	return clearVal;
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
		SetVertexShader( CompileShader( vs_4_0, main_vs()));
		SetGeometryShader( CompileShader( gs_4_0, main_gs()));
		SetPixelShader( CompileShader( ps_4_0, main_ps()));

		SetDepthStencilState( DSS_Disable, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}