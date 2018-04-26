#include "enginePool.fx"

float4 psBackground(QuadOutput input) : SV_TARGET
{
	return envMap.Sample(linearSampler, input.viewDir);
};

technique10 background
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsQuad() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psBackground() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

Texture2D showMap;

float4 psShow(QuadOutput input) : SV_TARGET
{
	return showMap.Sample(linearSampler, input.tex);
};

technique10 show
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsQuad() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psShow() ) );
		SetDepthStencilState( noDepthTestCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}