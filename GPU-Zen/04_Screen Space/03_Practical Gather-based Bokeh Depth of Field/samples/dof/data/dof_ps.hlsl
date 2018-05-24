SamplerState pointClampSampler: register(s0);
SamplerState linearClampSampler: register(s1);

Texture2D<float4> cocTexture: register(t0);
Texture2D<float4> cocNearBlurTexture: register(t1);
Texture2D<float4> colorTexture: register(t2);
Texture2D<float4> colorMulCoCFarTexture: register(t3);


cbuffer ConstantBuffer: register(b0)
{
	float2 pixelSize;
	float kernelScale;
	float padding;
}


struct PS_INPUT
{
	float4 position: SV_POSITION;
	float2 texCoord: TEXCOORD0;
};


struct PS_OUTPUT
{
	float4 near: SV_Target0;
	float4 far: SV_Target1;
};


static const float2 offsets[] =
{
	2.0f * float2(1.000000f, 0.000000f),
	2.0f * float2(0.707107f, 0.707107f),
	2.0f * float2(-0.000000f, 1.000000f),
	2.0f * float2(-0.707107f, 0.707107f),
	2.0f * float2(-1.000000f, -0.000000f),
	2.0f * float2(-0.707106f, -0.707107f),
	2.0f * float2(0.000000f, -1.000000f),
	2.0f * float2(0.707107f, -0.707107f),
	
	4.0f * float2(1.000000f, 0.000000f),
	4.0f * float2(0.923880f, 0.382683f),
	4.0f * float2(0.707107f, 0.707107f),
	4.0f * float2(0.382683f, 0.923880f),
	4.0f * float2(-0.000000f, 1.000000f),
	4.0f * float2(-0.382684f, 0.923879f),
	4.0f * float2(-0.707107f, 0.707107f),
	4.0f * float2(-0.923880f, 0.382683f),
	4.0f * float2(-1.000000f, -0.000000f),
	4.0f * float2(-0.923879f, -0.382684f),
	4.0f * float2(-0.707106f, -0.707107f),
	4.0f * float2(-0.382683f, -0.923880f),
	4.0f * float2(0.000000f, -1.000000f),
	4.0f * float2(0.382684f, -0.923879f),
	4.0f * float2(0.707107f, -0.707107f),
	4.0f * float2(0.923880f, -0.382683f),

	6.0f * float2(1.000000f, 0.000000f),
	6.0f * float2(0.965926f, 0.258819f),
	6.0f * float2(0.866025f, 0.500000f),
	6.0f * float2(0.707107f, 0.707107f),
	6.0f * float2(0.500000f, 0.866026f),
	6.0f * float2(0.258819f, 0.965926f),
	6.0f * float2(-0.000000f, 1.000000f),
	6.0f * float2(-0.258819f, 0.965926f),
	6.0f * float2(-0.500000f, 0.866025f),
	6.0f * float2(-0.707107f, 0.707107f),
	6.0f * float2(-0.866026f, 0.500000f),
	6.0f * float2(-0.965926f, 0.258819f),
	6.0f * float2(-1.000000f, -0.000000f),
	6.0f * float2(-0.965926f, -0.258820f),
	6.0f * float2(-0.866025f, -0.500000f),
	6.0f * float2(-0.707106f, -0.707107f),
	6.0f * float2(-0.499999f, -0.866026f),
	6.0f * float2(-0.258819f, -0.965926f),
	6.0f * float2(0.000000f, -1.000000f),
	6.0f * float2(0.258819f, -0.965926f),
	6.0f * float2(0.500000f, -0.866025f),
	6.0f * float2(0.707107f, -0.707107f),
	6.0f * float2(0.866026f, -0.499999f),
	6.0f * float2(0.965926f, -0.258818f),
};


float4 Near(float2 texCoord)
{
	float4 result = colorTexture.SampleLevel(pointClampSampler, texCoord, 0);
	
	for (int i = 0; i < 48; i++)
	{
		float2 offset = kernelScale * offsets[i] * pixelSize;
		result += colorTexture.SampleLevel(linearClampSampler, texCoord + offset, 0);
	}

	return result / 49.0f;
}


float4 Far(float2 texCoord)
{
	float4 result = colorMulCoCFarTexture.SampleLevel(pointClampSampler, texCoord, 0);
	float weightsSum = cocTexture.SampleLevel(pointClampSampler, texCoord, 0).y;
	
	for (int i = 0; i < 48; i++)
	{
		float2 offset = kernelScale * offsets[i] * pixelSize;
		
		float cocSample = cocTexture.SampleLevel(linearClampSampler, texCoord + offset, 0).y;
		float4 sample = colorMulCoCFarTexture.SampleLevel(linearClampSampler, texCoord + offset, 0);
		
		result += sample; // the texture is pre-multiplied so don't need to multiply here by weight
		weightsSum += cocSample;
	}

	return result / weightsSum;	
}


PS_OUTPUT PSMain(PS_INPUT input)
{
	PS_OUTPUT output;
	
	float cocNearBlurred = cocNearBlurTexture.SampleLevel(pointClampSampler, input.texCoord, 0).x;
	float cocFar = cocTexture.SampleLevel(pointClampSampler, input.texCoord, 0).y;
	float4 color = colorTexture.SampleLevel(pointClampSampler, input.texCoord, 0);

	if (cocNearBlurred > 0.0f)
		output.near = Near(input.texCoord);
	else
		output.near = color;

	if (cocFar > 0.0f)
		output.far = Far(input.texCoord);
	else
		output.far = 0.0f;

	return output;
}
