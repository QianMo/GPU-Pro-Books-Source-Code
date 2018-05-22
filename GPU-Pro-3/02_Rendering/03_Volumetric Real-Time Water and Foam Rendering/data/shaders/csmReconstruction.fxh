// -----------------------------------------------------------------------------
// -------------------------- CSM Reconstruction -------------------------------
// -----------------------------------------------------------------------------
float4 Ck_v(const float k) 
{
	return float4(PI * (2.0f*k - 1.0f),
				  PI * (2.0f*(k+1) - 1.0f),
				  PI * (2.0f*(k+2) - 1.0f),
				  PI * (2.0f*(k+3) - 1.0f));
}

float4 satArrayLookup(sampler2DARRAY sat, float3 uv, float size)
{
	size = clamp(size, shadowMapTexelSize, 1.0f);

	float4 color = tex2DARRAY(sat, float3(uv.xy + 0.5f*size, uv.z));			// LR
	color -= tex2DARRAY(sat, float3(uv.xy + float2(0.5f, -0.5f)*size, uv.z));	// UR
	color -= tex2DARRAY(sat, float3(uv.xy + float2(-0.5f, 0.5f)*size, uv.z));	// LL
	color += tex2DARRAY(sat, float3(uv.xy - 0.5f*size, uv.z));					// UL
	color /= size*size*shadowMapSizeSquared;

	return color;
}

float ShadowTerm(sampler2DARRAY sinMap, sampler2DARRAY cosMap, const float3 texCoord, const float filterSize)
{
	float4 d_v = float4(texCoord.z)-reconstructionOffset.y;
	
	float4 ck, sinVal, cosVal;

	float sum0 = 0.0f;
	float sum1 = 0.0f;

	for(float i=0; i<4.0f; ++i)
	{
		float idx = i * 4.0f;
		
		if (useMipMaps)
		{
			sinVal = tex2DARRAYlod(sinMap, float4(texCoord.xy, i, filterSize));
			cosVal = tex2DARRAYlod(cosMap, float4(texCoord.xy, i, filterSize));
		}
		else
		{
			sinVal = satArrayLookup(sinMap, float3(texCoord.xy, i), filterSize);
			cosVal = satArrayLookup(cosMap, float3(texCoord.xy, i), filterSize);
		}

		ck = Ck_v((i*4.0f)+1.0f);
	
		sum0 += cos(ck.x * d_v.x) / ck.x * sinVal.x;
		sum1 += sin(ck.x * d_v.x) / ck.x * cosVal.x;
		
		++idx;
		if(reconstructionOrder <= idx)
			break;

		sum0 += cos(ck.y * d_v.y) / ck.y * sinVal.y;
		sum1 += sin(ck.y * d_v.y) / ck.y * cosVal.y;

		++idx;
		if(reconstructionOrder <= idx)
			break;

		sum0 += cos(ck.z * d_v.z) / ck.z * sinVal.z;
		sum1 += sin(ck.z * d_v.z) / ck.z * cosVal.z;

		++idx;
		if(reconstructionOrder <= idx)
			break;

		sum0 += cos(ck.w * d_v.w) / ck.w * sinVal.w;
		sum1 += sin(ck.w * d_v.w) / ck.w * cosVal.w;

		++idx;
		if(reconstructionOrder <= idx)
			break;
	}

	float rec = 0.5 + 2.0 * (sum0 - sum1);

	// scale expansion
	return clamp(2.0f*rec, 0.0, 1.0);
}