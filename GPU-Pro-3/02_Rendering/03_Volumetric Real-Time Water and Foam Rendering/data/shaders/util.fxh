
struct FragmentShaderOutput
{	
	float4 color : COLOR;
};

struct PositionTexCoordShaderInput 
{
	float4 position : POSITION;
	float2 texCoord : TEXCOORD0;
};

PositionTexCoordShaderInput PassThroughVP(PositionTexCoordShaderInput IN)
{
	PositionTexCoordShaderInput OUT;

	OUT.position = IN.position;
	OUT.texCoord = IN.texCoord;

	return OUT;
}