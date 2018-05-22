matrix WorldViewProj;
float Alpha;

struct LinkedListEntryDepthAlphaNext
{
	float depth;
	float alpha;
	int next;
};

#include "DeepShadowMapGlobal.hlsl"

RWStructuredBuffer<StartElementBufEntry> StartElementBuf;

RWStructuredBuffer<LinkedListEntryDepthAlphaNext> LinkedListBufDAN;
StructuredBuffer<LinkedListEntryDepthAlphaNext> LinkedListBufDANRO;

RWStructuredBuffer<LinkedListEntryWithPrev> LinkedListBufWP;
RWStructuredBuffer<LinkedListEntryNeighbors> NeighborsBuf;

struct VS_IN
{
	float4 pos : POSITION;
	float3 norm : NORMAL;
};

struct PS_IN
{
	float4 pos : SV_POSITION;
};

PS_IN vs_main(VS_IN input)
{
	PS_IN output = (PS_IN)0;
	output.pos = mul(input.pos, WorldViewProj);
	return output;
}

void ps_main(PS_IN input)
{	
	// do nothing
}

[numthreads(16, 8, 1)]
void cs_sort(uint3 DTid : SV_DispatchThreadID)
{
	// do nothing
}

[numthreads(16, 8, 1)]
void cs_link(uint3 DTid : SV_DispatchThreadID)
{
	// do nothing
}

RasterizerState DisableCullingBias
{
    CullMode = NONE;
	SlopeScaledDepthBias = 25.0f;
};

technique11 Render
{
	pass P0
	{
		SetRasterizerState(DisableCullingBias);  
		SetVertexShader(CompileShader(vs_5_0, vs_main()));
		SetPixelShader (CompileShader(ps_5_0, ps_main()));
	}
}

technique11 Sort
{
	pass P0
	{
		SetComputeShader(CompileShader(cs_5_0, cs_sort()));
	}
}

technique11 Link
{
	pass P0
	{
		SetComputeShader(CompileShader(cs_5_0, cs_link()));
	}
}