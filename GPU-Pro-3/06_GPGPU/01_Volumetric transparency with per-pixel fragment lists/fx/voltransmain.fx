#include "renderstates.fx"
#include "basic.fx"
#include "voltransdefer.fx"
#include "voltransdata.fx"

#include "volparticles.fx"

#include "voltransshadow.fx"
#include "voltransstore.fx"
#include "voltranssort.fx"
#include "voltransfluid.fx"

technique11 volumetricTransparency
{
	pass defer
	{
		SetVertexShader ( CompileShader( vs_5_0, vsTbnTrafo() ) );
		SetGeometryShader ( NULL );
		SetRasterizerState( defaultRasterizer );
		SetPixelShader( CompileShader( ps_5_0, psDefer() ) );
		SetDepthStencilState( defaultCompositor, 0 );
		SetBlendState( defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff );
	}
	pass background
	{
		SetVertexShader ( CompileShader( vs_5_0, vsQuad() ) );
		SetGeometryShader ( NULL );
		SetRasterizerState( noCullRasterizer );
		SetPixelShader( CompileShader( ps_5_0, psDeferBackground() ) );
		SetDepthStencilState( noDepthWriteCompositor, 0 );
		SetBlendState( defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff );
	}
	pass storeFragments
	{
		SetVertexShader ( CompileShader( vs_5_0, vsTrafo() ) );
		SetGeometryShader ( NULL );
		SetRasterizerState( noCullRasterizer );
		SetPixelShader( CompileShader( ps_5_0, psStoreFragments() ) );
		SetDepthStencilState( noDepthTestCompositor, 0 );
		SetBlendState( addBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff );
	}
	pass storeFluidFragments
	{
		SetVertexShader ( CompileShader( vs_5_0, vsFluid() ) );
		SetGeometryShader ( NULL );
		SetRasterizerState( noCullRasterizer );
		SetPixelShader( CompileShader( ps_5_0, psStoreFragments() ) );
		SetDepthStencilState( noDepthTestCompositor, 0 );
		SetBlendState( addBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff );
	}
	pass storeShadowFragments
	{
		SetVertexShader ( CompileShader( vs_5_0, vsShadowVolume() ) );
		SetGeometryShader ( CompileShader( gs_5_0, gsShadowVolume() ) );
		SetRasterizerState( noCullRasterizer );
		SetPixelShader( CompileShader( ps_5_0, psStoreFragments() ) );
		SetDepthStencilState( noDepthTestCompositor, 0 );
		SetBlendState( addBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff );
	}
	pass sortAndRender
	{
		SetVertexShader ( CompileShader( vs_5_0, vsQuad() ) );
		SetGeometryShader ( NULL );
		SetRasterizerState( defaultRasterizer );
		SetPixelShader( CompileShader( ps_5_0, psSortAndRender() ) );
		SetDepthStencilState( noDepthTestCompositor, 0 );
		SetBlendState( defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff );
	}
}