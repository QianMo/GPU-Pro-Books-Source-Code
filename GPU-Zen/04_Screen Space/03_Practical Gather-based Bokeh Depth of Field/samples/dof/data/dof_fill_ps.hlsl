SamplerState pointClampSampler: register(s0);
SamplerState linearClampSampler: register(s1);

Texture2D<float4> cocTexture: register(t0);
Texture2D<float4> cocBlurTexture: register(t1);
Texture2D<float4> dofNearTexture: register(t2);
Texture2D<float4> dofFarTexture: register(t3);


cbuffer ConstantBuffer: register(b0)
{
	float2 pixelSize;
	float2 padding;
}


struct PS_INPUT
{
	float4 position: SV_POSITION;
	float2 texCoord: TEXCOORD0;
};


struct PS_OUTPUT
{
	float4 nearFill: SV_Target0;
	float4 farFill: SV_Target1;
};


PS_OUTPUT PSMain(PS_INPUT input)
{
	PS_OUTPUT output;

	float cocNearBlurred = cocBlurTexture.SampleLevel(pointClampSampler, input.texCoord, 0).x;
	float cocFar = cocTexture.SampleLevel(pointClampSampler, input.texCoord, 0).y;

	output.nearFill = dofNearTexture.SampleLevel(pointClampSampler, input.texCoord, 0);
	output.farFill = dofFarTexture.SampleLevel(pointClampSampler, input.texCoord, 0);	

	if (cocNearBlurred > 0.0f)
	{
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				float2 sampleTexCoord = input.texCoord + float2(i, j)*pixelSize;
				float4 sample = dofNearTexture.SampleLevel(pointClampSampler, sampleTexCoord, 0);
				output.nearFill = max(output.nearFill, sample);
			}
		}
	}

	if (cocFar > 0.0f)
	{
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				float2 sampleTexCoord = input.texCoord + float2(i, j)*pixelSize;
				float4 sample = dofFarTexture.SampleLevel(pointClampSampler, sampleTexCoord, 0);
				output.farFill = max(output.farFill, sample);
			}
		}
	}

	return output;
}
