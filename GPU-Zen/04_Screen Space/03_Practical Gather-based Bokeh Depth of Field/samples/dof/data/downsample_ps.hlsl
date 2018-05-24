SamplerState pointClampSampler: register(s0);
SamplerState linearClampSampler: register(s1);

Texture2D<float4> colorTexture: register(t0);
Texture2D<float4> cocTexture: register(t1);


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
	float4 color: SV_Target0;
	float4 colorMulCOCFar: SV_Target1;
	float4 coc: SV_Target2;
};


PS_OUTPUT PSMain(PS_INPUT input)
{
	PS_OUTPUT output;
	
	//
	
	float2 texCoord00 = input.texCoord + float2(-0.25f, -0.25f)*pixelSize;
	float2 texCoord10 = input.texCoord + float2( 0.25f, -0.25f)*pixelSize;
	float2 texCoord01 = input.texCoord + float2(-0.25f,  0.25f)*pixelSize;
	float2 texCoord11 = input.texCoord + float2( 0.25f,  0.25f)*pixelSize;
	
	//

	float4 color = colorTexture.SampleLevel(linearClampSampler, input.texCoord, 0);
	float4 coc = cocTexture.SampleLevel(pointClampSampler, texCoord00, 0);
	
	// custom bilinear filtering of color weighted by coc far
	
	float cocFar00 = cocTexture.SampleLevel(pointClampSampler, texCoord00, 0).y;
	float cocFar10 = cocTexture.SampleLevel(pointClampSampler, texCoord10, 0).y;
	float cocFar01 = cocTexture.SampleLevel(pointClampSampler, texCoord01, 0).y;
	float cocFar11 = cocTexture.SampleLevel(pointClampSampler, texCoord11, 0).y;

	float weight00 = 1000.0f;
	float4 colorMulCOCFar = weight00 * colorTexture.SampleLevel(pointClampSampler, texCoord00, 0);
	float weightsSum = weight00;
	
	float weight10 = 1.0f / (abs(cocFar00 - cocFar10) + 0.001f);
	colorMulCOCFar += weight10 * colorTexture.SampleLevel(pointClampSampler, texCoord10, 0);
	weightsSum += weight10;
	
	float weight01 = 1.0f / (abs(cocFar00 - cocFar01) + 0.001f);
	colorMulCOCFar += weight01 * colorTexture.SampleLevel(pointClampSampler, texCoord01, 0);
	weightsSum += weight01;
	
	float weight11 = 1.0f / (abs(cocFar00 - cocFar11) + 0.001f);
	colorMulCOCFar += weight11 * colorTexture.SampleLevel(pointClampSampler, texCoord11, 0);
	weightsSum += weight11;

	colorMulCOCFar /= weightsSum;
	colorMulCOCFar *= coc.y;

	//

	output.color = color;
	output.colorMulCOCFar = colorMulCOCFar;
	output.coc = coc;
	
	//

	return output;
}
