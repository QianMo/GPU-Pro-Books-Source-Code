#ifdef MD_D3D9

#define MD_TECHNIQUE technique
#define MD_VS_TARGET vs_3_0
#define MD_PS_TARGET ps_3_0
// to be placed near params due to ignored shared cbuffer syntax
#define MD_SHAREDP shared

float4 f4tex2D( Texture2D tex, sampler2D samp, float2 texc )
{
	return tex2D( samp, texc );
}

float4 f4texCUBE( TextureCube tex, samplerCUBE samp, float3 texc )
{
	return texCUBE( samp, texc );
}

float4 f4tex2Dgrad( Texture2D tex, sampler2D samp, float2 texc, float2 dx, float2 dy )
{
	return tex2Dgrad( samp, texc, dx, dy );
}

float4 f4texCUBEgrad( TextureCube tex, samplerCUBE samp, float3 texc, float3 dx, float3 dy )
{
	return texCUBEgrad( samp, texc, dx, dy );
}

#elif defined( MD_D3D10 )

#define MD_TECHNIQUE technique10
#define MD_VS_TARGET vs_4_0
#define MD_PS_TARGET ps_4_0
#define MD_SHAREDP

float4 f4tex2D( Texture2D tex, SamplerState samp, float2 texc )
{
	return tex.Sample( samp, texc );
}

float4 f4texCUBE( TextureCube tex, SamplerState samp, float3 texc )
{
	return tex.Sample( samp, texc );
}

float4 f4tex2Dgrad( Texture2D tex, SamplerState samp, float2 texc, float2 dx, float2 dy )
{
	return tex.SampleGrad( samp, texc, dx, dy );
}

float4 f4texCUBEgrad( TextureCube tex, SamplerState samp, float3 texc, float3 dx, float3 dy )
{
	return tex.SampleGrad( samp, texc, dx, dy );
}


#endif