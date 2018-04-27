//

DepthStencilState disableDepth
{
    DepthEnable = FALSE;
    DepthWriteMask = ZERO;
};

DepthStencilState enableDepth
{
    DepthEnable = TRUE;
    DepthWriteMask = ALL;
};



cbuffer perFrame
{
	float4x4 	matWVP;
	float4x3 	matArrBones[128];
	float		scale = 1.1;
};

uint main_vs( uint id : SV_VertexID ) : ID
{
	return id;
}

struct GSOut
{
	float4 pos 	: SV_Position;
	float4 colr : COLOR;
};

typedef GSOut PSIn;

[maxvertexcount(6)]
// reading primitive ID directly with idle vertex shader now crashses ( it didn't a year ago I swar ;] )
void main_gs( point uint ia[1] : ID, inout LineStream<GSOut> os)
{
	uint id = ia[0];

	float4x3 boneTrans = matArrBones[id];
	
	float3 start = mul(float4(0,0,0,1), boneTrans);

	float3 axes[3] = { 	mul(float3(1,0,0), (float3x3)boneTrans),
						mul(float3(0,1,0), (float3x3)boneTrans),
						mul(float3(0,0,1), (float3x3)boneTrans) };

	float4 axesColrs[3] = { float4(1,0,0,1),
							float4(0,1,0,1),
							float4(0,0,1,1)
							};

	float4 psStart = mul(float4(start,1), matWVP); // projection space start ;]

	[unroll]
	for(uint i=0;i<3;i++)
	{
		GSOut o;
		o.colr = axesColrs[i];		

		o.pos = psStart;
		os.Append(o);

		o.pos = mul(float4(start+axes[i]*scale, 1), matWVP);
		os.Append(o);

		os.RestartStrip();
	}
}

float4 main_ps(PSIn pi) : SV_Target0
{
	return pi.colr;
}

technique10 main
{
	pass
	{
		SetVertexShader(CompileShader(vs_4_0, main_vs()));
		SetGeometryShader(CompileShader(gs_4_0, main_gs()));
		SetPixelShader(CompileShader(ps_4_0, main_ps()));
		SetDepthStencilState(disableDepth, 0);
	}
	pass
	{
		SetDepthStencilState(enableDepth, 0);
	}	

}