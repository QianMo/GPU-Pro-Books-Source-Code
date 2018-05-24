SamplerState pointClampSampler: register(s0);
SamplerState linearClampSampler: register(s1);

Texture2D<float4> colorTexture: register(t0);
Texture2D<float4> cocTexture: register(t1);
Texture2D<float4> cocTexture_x4: register(t2);
Texture2D<float4> cocBlurTexture_x4: register(t3);
Texture2D<float4> dofNearTexture_x4: register(t4);
Texture2D<float4> dofFarTexture_x4: register(t5);


cbuffer ConstantBuffer: register(b0)
{
	float2 pixelSize;
	float blend;
	float padding;
}


struct PS_INPUT
{
	float4 position: SV_POSITION;
	float2 texCoord: TEXCOORD0;
};


float4 PSMain(PS_INPUT input): SV_Target
{
	float4 result = colorTexture.SampleLevel(pointClampSampler, input.texCoord, 0);

	// far field
	{
		float2 texCoord00 = input.texCoord;
		float2 texCoord10 = input.texCoord + float2(pixelSize.x, 0.0f);
		float2 texCoord01 = input.texCoord + float2(0.0f, pixelSize.y);
		float2 texCoord11 = input.texCoord + float2(pixelSize.x, pixelSize.y);

		float cocFar = cocTexture.SampleLevel(pointClampSampler, input.texCoord, 0).y;
		float4 cocsFar_x4 = cocTexture_x4.GatherGreen(pointClampSampler, texCoord00).wzxy;
		float4 cocsFarDiffs = abs(cocFar.xxxx - cocsFar_x4);

		float4 dofFar00 = dofFarTexture_x4.SampleLevel(pointClampSampler, texCoord00, 0);
		float4 dofFar10 = dofFarTexture_x4.SampleLevel(pointClampSampler, texCoord10, 0);
		float4 dofFar01 = dofFarTexture_x4.SampleLevel(pointClampSampler, texCoord01, 0);
		float4 dofFar11 = dofFarTexture_x4.SampleLevel(pointClampSampler, texCoord11, 0);

		float2 imageCoord = input.texCoord / pixelSize;
		float2 fractional = frac(imageCoord);
		float a = (1.0f - fractional.x) * (1.0f - fractional.y);
		float b = fractional.x * (1.0f - fractional.y);
		float c = (1.0f - fractional.x) * fractional.y;
		float d = fractional.x * fractional.y;

		float4 dofFar = 0.0f;
		float weightsSum = 0.0f;

		float weight00 = a / (cocsFarDiffs.x + 0.001f);
		dofFar += weight00 * dofFar00;
		weightsSum += weight00;

		float weight10 = b / (cocsFarDiffs.y + 0.001f);
		dofFar += weight10 * dofFar10;
		weightsSum += weight10;

		float weight01 = c / (cocsFarDiffs.z + 0.001f);
		dofFar += weight01 * dofFar01;
		weightsSum += weight01;

		float weight11 = d / (cocsFarDiffs.w + 0.001f);
		dofFar += weight11 * dofFar11;
		weightsSum += weight11;

		dofFar /= weightsSum;

		result = lerp(result, dofFar, blend * cocFar);
	}

	// near field
	{
		float cocNear = cocBlurTexture_x4.SampleLevel(linearClampSampler, input.texCoord, 0).x;
		float4 dofNear = dofNearTexture_x4.SampleLevel(linearClampSampler, input.texCoord, 0);

		result = lerp(result, dofNear, blend * cocNear);
	}
	
	return result;
}
