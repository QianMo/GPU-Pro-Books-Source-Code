#include "enginePool.fx"

struct DeferredFragment
{
	float4 geometry : SV_TARGET0;
	float4 brdf : SV_TARGET1;
};

DeferredFragment psDeferring(TrafoOutput input)
{
	DeferredFragment output;
	float3 normal = normalize(input.normal);

//	float3 relfDir = reflect(normalize(input.worldPos - eyePosition), normal);
//	normal = relfDir;

	output.geometry.xyz = normal;
	output.geometry.w = length(input.worldPos - eyePosition);
	output.brdf = kdMap.Sample(linearSampler, input.tex);
	output.brdf.w = 1;
	return output;
}

technique10 deferring
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsTrafo() ) );
        SetGeometryShader( NULL );        
		SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psDeferring() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

DeferredFragment psBackgroundDeferring(QuadOutput input)
{
	DeferredFragment output;
	output.brdf = envMap.Sample(linearSampler, input.viewDir);
	output.geometry = 0;
	return output;
};


technique10 backgroundDeferring
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsQuad() ) );
        SetGeometryShader( NULL );
		SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psBackgroundDeferring() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}


