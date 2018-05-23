#ifndef BE_SKYCOLOR_H
#define BE_SKYCOLOR_H

float3 EvaluateSkyColor(float3 normalizedDirection,
	uniform float4 skyColor, uniform float4 hazeColor,
	uniform bool bEnableGround = true, uniform float4 groundColor = 0.0f)
{
	float skyBlend = saturate( (normalizedDirection.y - hazeColor.w) / (skyColor.w - hazeColor.w) );
	float3 skyMix = lerp(hazeColor.xyz, skyColor.xyz, skyBlend);
	
	float3 color = skyMix;

	if (bEnableGround)
	{
		float groundBlend = saturate( normalizedDirection.y * 1000 );
		float hazeBlend = saturate( (normalizedDirection.y - groundColor.w) / -groundColor.w );

		float3 groundMix = lerp(groundColor.xyz, hazeColor.xyz, hazeBlend);
		color = lerp(groundMix, color, groundBlend);
	}

	return color;
}

#endif