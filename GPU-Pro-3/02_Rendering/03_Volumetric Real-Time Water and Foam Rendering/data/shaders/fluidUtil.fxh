
// convert [0,1] uv coords and eye-space Z to eye-space position
float3 uvToEye(float2 uv, float eyeZ)
{
	uv = uv * invViewport;
	uv = uv * float2(-2.0f, -2.0f) - float2(-1.0f, -1.0f);
	return float3(uv * invFocalLength * eyeZ, eyeZ);
}

// good approximation to uvToEye(uv + uvdiff, eyeZ2) - uvToEye(uv, eyeZ)
float3 uvToEyeD(float2 uvdiff, float eyeZ2, float eyeZ)
{
	return float3(invCamera*uvdiff*eyeZ, eyeZ2 - eyeZ);
}

// calculate eye space position depending on texture coordinates
float3 getEyeSpacePos(float2 texCoord)
{
	float eyeZ = texRECT(depthMap, texCoord).x;
	return uvToEye(texCoord, eyeZ);
}

// depth interpolation
float2 ipdepth(float ep1m, float ep1c, float ep1p, float f1)
{
	float2 d1 = float2(ep1c - ep1m, (ep1c + ep1m)*0.5f);
	float2 d2 = float2(ep1p - ep1c, (ep1p + ep1c)*0.5f);
	return lerp(d1, d2, f1);
}

// normal interpolation
float4 ipnormal2(float2 tc, float2 d1, float f1)
{
	float ep1m  = texRECT(depthMap, tc - d1).x;
	float ep1c  = texRECT(depthMap, tc).x;
	float ep1p  = texRECT(depthMap, tc + d1).x;

	float2 dv1 = ipdepth(ep1m, ep1c, ep1p, f1);

	float4 n1 = float4(uvToEyeD(d1, ep1c + dv1.x, ep1c), dv1.y);

	return n1;
}

// calculate soft shadow term
float GetShadowTerm(float3 viewSpacePosition)
{
	float shadowTerm = 1.0f;

	float4 smTexCoord;
	float zEyeSpace = mul(shadowMapLinearTextureMatrix, float4(viewSpacePosition.xyz, 1.0f)).z;
	smTexCoord = mul(shadowMapTextureMatrix, float4(viewSpacePosition.xyz, 1.0f));
	smTexCoord.z = (zEyeSpace + clipPlanes.x) / clipPlanes.z;

	float2 uv = smTexCoord.xy/smTexCoord.w;

	if (useMipMaps)
		shadowTerm = ShadowTerm(csmSinMap, csmCosMap, float3(uv, smTexCoord.z), log2(0.05f*shadowMapSize));
	else
		shadowTerm = ShadowTerm(csmSinMap, csmCosMap, float3(uv, smTexCoord.z), 0.0001f*shadowMapSize);

	return shadowTerm;
}

// shade a pixel based on partial derivatives and eye position
float4 shade(float3 _ddx, float3 _ddy, float3 eyeSpacePos, float2 noiseNormal, float2 texCoord, float shadowTerm)
{
	// normal calculation
	float3 normal;
	_ddx.z += noiseNormal.x;
	_ddy.z += noiseNormal.y;
	normal = cross(_ddx.xyz, _ddy.xyz);
	normal = normalize(normal);

	// surface properties (v = view vector, h = half angle vector)
	float3 lightDir = normalize(lightPosEyeSpace.xyz - eyeSpacePos);
	float3 v = normalize(-eyeSpacePos);
	float3 h = normalize(lightDir + v);
	float specular = pow(max(0.0, dot(normal, h)), fluidShininess)*shadowTerm;

	// disable specular for pixels not part of the fluid surface (background pixels with far away depth)
	specular *= step(-9999.0f, eyeSpacePos.z);

	// bias, scale, and power = user defined parameters to tune the Fresnel Term
	float fresnelTerm = fresnelBias + fresnelScale*pow(1.0 - max(0.0, dot(normal, v)), fresnelPower);

	// cubemap reflection
	float4 c_reflect = texCUBE(cubeMap, mul((float3x3)invView, reflect(-v, normal)));

	// water and foam thickness
	// t_wb = t_water.x
	// t_wf = t_water.y
	float4 t_water = texRECT(thicknessMap, texCoord.xy*lowResFactor);
	// t_f = t_foam.x
	// t_ff = t_foam.y
	float4 t_foam = texRECT(foamThicknessMap, texCoord.xy*lowResFactor);

	// calculate c_fluid terms
	float4 c_fluid_wb = baseColor * exp(-t_water.x*falloffScale*colorFalloff);
	float4 c_fluid_wf = baseColor * exp(-t_water.y*falloffScale*colorFalloff);

	// attenuation factors (including user defined falloff scales)
	float att_wb = saturate(c_fluid_wb.w);
	float att_foam = saturate(exp(-t_foam.x*foamFalloffScale));
	float att_ff = saturate(exp(-t_foam.y*foamFrontFalloffScale));
	float att_wf = saturate(c_fluid_wf.w);

	// composition
	float4 c_background = texRECT(sceneMap, (texCoord.xy * lowResFactor) + (normal.xy * (t_water.x + t_water.y) * thicknessRefraction));
	float3 c_wb = lerp(c_fluid_wb.xyz, c_background.xyz, att_wb);
	float3 c_foam = lerp(foamFrontColor, foamBackColor, att_ff).xyz;

	// apply shadow term to foam color
	float3 Ka = float3(0.4f, 0.4f, 0.4f);
	c_foam = Ka*c_foam*(1.0f-shadowTerm) + c_foam*shadowTerm;

	// composition (cont.)
	float3 c_f  = lerp(c_foam.xyz, c_wb.xyz, att_foam);
	float3 c_wf = lerp(c_fluid_wf.xyz, c_f.xyz, att_wf);

	// calculate factor to suppress specular highlights if foam is the frontmost visual element
	float surfPropWeight = saturate(1.0f-att_wf);

	// combine with fresnel and specular highlight
	float4 surfaceColor = float4(lerp(c_wf.xyz, c_reflect.xyz, min(fresnelTerm, surfPropWeight)) + fluidSpecularColor.xyz*specular*surfPropWeight, 1.0f);

	if (renderMode <= 0.0f)
		return float4(c_background.xyz, 1.0f);
	else if (renderMode <= 1.0f)
		return float4(c_wb.xyz, 1.0f);
	else if (renderMode <= 2.0f)
		return float4(c_f.xyz, 1.0f);
	else if (renderMode <= 3.0f)
		return float4(c_wf.xyz, 1.0f);
	else
		return surfaceColor;
}

// used for rendering border of the fluid
float4 sample(float2 tc, float2 noiseNormal, float2 texCoord, float shadowTerm)
{
	float3 eyeSpacePos = getEyeSpacePos(tc);
	
	float3 _ddx = getEyeSpacePos(tc + float2(1.0f, 0.0f)) - eyeSpacePos;
	float3 _ddx2 = eyeSpacePos - getEyeSpacePos(tc + float2(-1.0f, 0.0f));
	if(abs(_ddx2.z) < abs(_ddx.z))
		_ddx = _ddx2;

	float3 _ddy = getEyeSpacePos(tc + float2(0.0f, 1.0f)) - eyeSpacePos;
	float3 _ddy2 = eyeSpacePos - getEyeSpacePos(tc + float2(0.0f, -1.0f));
	if(abs(_ddy2.z) < abs(_ddy.z))
		_ddy = _ddy2;

	return shade(_ddx, _ddy, eyeSpacePos, noiseNormal, texCoord, shadowTerm);
}
