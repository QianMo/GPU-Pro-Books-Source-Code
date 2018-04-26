#include "enginePool.fx"

TrafoOutput
	vsBalls(TrafoInput input, uint iid : SV_INSTANCEID)
{
	float4 occ = occluderSphereSetArray.Load(uint3(iid, 1, 0));
	input.pos.xyz *= occ.w;
	input.pos.xyz += occ.xyz;

	TrafoOutput output = (TrafoOutput)0;
	output.pos = mul(input.pos, modelViewProjMatrix);

	output.worldPos = mul(input.pos, modelMatrix);

	output.normal = mul(modelMatrixInverse, float4(input.normal.xyz, 0.0));
	output.tex = input.tex;
	
	return output;
}

float4 psBalls(TrafoOutput input) : SV_TARGET
{
	float3 normal = normalize(input.normal);
	return normal.y;
}

technique10 balls
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsBalls() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psBalls() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}