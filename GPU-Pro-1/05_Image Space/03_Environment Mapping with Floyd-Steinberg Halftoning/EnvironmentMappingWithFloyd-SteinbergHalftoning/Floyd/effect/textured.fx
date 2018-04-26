#include "enginePool.fx"

float4 psBasic(TrafoOutput input) : SV_TARGET
{
	float3 normal = normalize(input.normal);
	return (abs(normal.y) * 0.9 + 0.1) * kdMap.Sample(linearSampler, input.tex);
}

technique10 basic
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsTrafo() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psBasic() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

float4 psNormalMapped(TrafoOutput input) : SV_TARGET
{
	float3 normal = normalMap.Sample(linearSampler, input.tex);
	normal -= float3(0.5, 0.5, 0.5);
	normal = normalize(normal);
	float3 reflectDir = reflect(eyePosition - input.worldPos, normal);

	return envMap.Sample(linearSampler, reflectDir)*0.15 + 
		saturate(dot(normal, float3(0, 1, 0)) * float4(0.5, 0.5, 0.4, 0.5));
//	return input.normal.y;
//	return input.tex.xxyy;
//	input.tex.y = - input.tex.y;
//	input.tex.x = - input.tex.x;
	
//	return abs(normalMap.Sample(linearSampler, input.tex));
}

technique10 normalMapped
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsTrafo() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psNormalMapped() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}


struct TestInput
{
    float4 pos			: POSITION;
    float3 normal		: NORMAL;
    float2 tex			: TEXCOORD0;
//	float4 a			: BLENDWEIGHT;
//	float4 b			: BLENDINDICES;
};

struct TestOutput
{
    float4 pos			: SV_POSITION;
    float3 normal		: TEXCOORD2;
    float2 tex			: TEXCOORD0;
    float4 worldPos		: TEXCOORD1;
//	float4 a			: BLENDWEIGHT;
//	float4 b			: BLENDINDICES;
};

TestOutput
	vsTest(TestInput input)
{
	input.pos.y = 0;

	TestOutput output = (TestOutput)0;
	output.pos = mul(input.pos, modelViewProjMatrix);

	output.worldPos = mul(input.pos, modelMatrix);
	
	output.normal = mul(modelMatrixInverse, float4(input.normal.xyz, 0.0));
	output.tex = input.tex;

//	output.a = input.a;
//	output.b = input.b;
	
	return output;
}

float3x3 cutMatrix = {
	float3(0.9, 1.2, 1),
	float3(-0.2, 0.4, -1),
	float3(0, 1.2, 1)};

float4 psTest(TestOutput input) : SV_TARGET
{
	float2 tc = input.tex;

	float3 tp = mul(cutMatrix, float3(tc, 1));

	return frac(tp.xyzz);
//	return frac(tc).xxyy;
	return kdMap.Sample(linearSampler, tc);
}

technique10 test
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsTest() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psTest() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}
