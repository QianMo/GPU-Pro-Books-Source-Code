#include "enginePool.fx"

Texture2D geometryViewportMap;
Texture2D brdfViewportMap;

float4 psDeferred(QuadOutput input) : SV_TARGET
{
	float4 geometry = geometryViewportMap.Load(int3(input.pos.xy, 0));
	float4 brdf = brdfViewportMap.Load(int3(input.pos.xy, 0));

	input.viewDir = normalize(input.viewDir);
	float3 worldPos = eyePosition + input.viewDir * geometry.w;

	return geometry.w / 100;

/*	float3 lightDiff = spotLights[0].position - worldPos;
	float3 lightDir = normalize(lightDiff);
	float3 lighting = spotLights[0].peakRadiance * max(0, dot(geometry.xyz, lightDir)) * 
		pow(max(0,dot(-lightDir, spotLights[0].direction)), spotLights[0].focus) / dot(lightDiff, lightDiff);

	return brdf ;//* geometry.y;//float4(lighting, 1) * brdf;*/


};

technique10 deferred
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsQuad() ) );
        SetGeometryShader( NULL );
		SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psDeferred() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}