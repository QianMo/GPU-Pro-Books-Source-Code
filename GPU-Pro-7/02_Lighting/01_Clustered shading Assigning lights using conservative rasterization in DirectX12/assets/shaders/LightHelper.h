#ifndef LIGHTHELPER
#define LIGHTHELPER

static const float PI = 3.14159265f;

struct PointLight
{
	float4 pos;
	float4 color;
};

struct SpotLight
{
	float4 pos;
	float4 color;
    float4 dir_angle;
};

float3 PointLightCalc(float4 surfacePos, float4 surfaceNormal, PointLight L)
{
    float3 litColor = float3(0.0f, 0.0f, 0.0f);

    // The vector from the surface to the light
    float3 lightVec = L.pos.xyz - surfacePos.xyz;

    float d = length(lightVec);

	lightVec = normalize(lightVec);

    // Return if outside range
	if (d < L.pos.w)
	{
		float x = d / L.pos.w;
		float attenuation = 1.0f - x;

		litColor = L.color.xyz * saturate(dot(lightVec, surfaceNormal.xyz)) * attenuation;
	}

	return litColor;
}

float3 SpotLightCalc(float4 surfacePos, float4 surfaceNormal, SpotLight L)
{
	float3 litColor = float3(0.0f, 0.0f, 0.0f);

	// The vector from the surface to the light
	float3 lightVec = L.pos.xyz - surfacePos.xyz;
    float d = length(lightVec);
	lightVec = normalize(lightVec);
	float cosine_cone_angle = cos(L.dir_angle.w);
	float cosine_current_cone_angle = dot(-lightVec, L.dir_angle.xyz);

    // Return if outside range
	if (d < L.pos.w && cosine_current_cone_angle > cosine_cone_angle)
	{
		float radial_attenuation = (cosine_current_cone_angle - cosine_cone_angle) / (1.0f - cosine_cone_angle);
		radial_attenuation = radial_attenuation * radial_attenuation;

		float x = d / L.pos.w;
		float attenuation = pow(1.0f - x, 0.5f);

		litColor = L.color.xyz * saturate(dot(lightVec, surfaceNormal.xyz)) * attenuation *	radial_attenuation;
	}

	return litColor;
}
#endif
