
float4x4 WorldViewProjection	: WORLDVIEWPROJECTION;
float4x4 WorldViewIT			: WORLDVIEWINVERSETRANSPOSE;
float4	LightDirection			: LIGHTDIRWORLD0;

struct VS_INPUT
{
	float3 Vertex		: POSITION;
	float3 Normal		: NORMAL;
};

struct VS_OUTPUT
{
	float4 Position		: SV_POSITION;
	float  Intensity	: LIGHTINTENSITY;
};


VS_OUTPUT VertShader(VS_INPUT In)
{
	VS_OUTPUT Out;
	
	Out.Position = mul(float4(In.Vertex, 1.0), WorldViewProjection);
	
	float3 normal =  mul(float4(In.Normal, 1.0), WorldViewIT);
	Out.Intensity = abs( dot(float3(0.0, 0.0, 1.0), normalize(normal)) );
	
	return Out;
}

float4 PixShader(VS_OUTPUT In) : SV_Target
{
	float4 color;
	
	if (In.Intensity > 0.95)
		color = float4(1.0,0.5,0.5,1.0);
	else if (In.Intensity > 0.5)
		color = float4(0.6,0.3,0.3,1.0);
	else if (In.Intensity > 0.25)
		color = float4(0.4,0.2,0.2,1.0);
	else
		color = float4(0.2,0.1,0.1,1.0);

	return color;	
}

technique10 ToonEffect
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_4_0, VertShader() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, PixShader() ) );
    }
}







