SamplerState pointClampSampler: register(s0);
SamplerState linearClampSampler: register(s1);

Texture2D<float4> depth16Texture: register(t0);
Texture2D<float4> ssaoTexture: register(t1);


static float gaussWeightsSigma1[7] =
{
	0.00598f,
	0.060626f,
	0.241843f,
	0.383103f,
	0.241843f,
	0.060626f,
	0.00598f
};

static float gaussWeightsSigma3[7] =
{
	0.106595f,
	0.140367f,
	0.165569f,
	0.174938f,
	0.165569f,
	0.140367f,
	0.106595f
};


cbuffer ConstantBuffer: register(b0)
{
	float2 pixelOffset;
	float2 padding;
}


struct PS_INPUT
{
	float4 position: SV_POSITION;
	float2 texCoord: TEXCOORD0;
};


float4 PSMain(PS_INPUT input): SV_Target0
{
	float sum = 0.0f;
	float weightsSum = 0.0f;

	float depth = depth16Texture.Sample(pointClampSampler, input.texCoord).x;

	for (int i = -3; i <= 3; i++)
	{
		float2 sampleTexCoord = input.texCoord + i*pixelOffset;
		float sampleDepth = depth16Texture.Sample(pointClampSampler, sampleTexCoord).x;

		float depthsDiff = 0.1f * abs(depth - sampleDepth);
		depthsDiff *= depthsDiff;
		float weight = 1.0f / (depthsDiff + 0.001f);
		weight *= gaussWeightsSigma3[3 + i];

		sum += weight * ssaoTexture.Sample(pointClampSampler, sampleTexCoord);
		weightsSum += weight;
	}

	return sum / weightsSum;
}
