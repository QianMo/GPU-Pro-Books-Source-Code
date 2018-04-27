//

cbuffer perFrame
{
	float4x4 matWVP;
};

float4 main_vs( float4 pos: POSITION ) : SV_Position
{
	return mul(pos, matWVP);
}

void main_ps()
{
}

technique10 main
{
	pass
	{
		SetVertexShader(CompileShader(vs_4_0, main_vs()));
		SetGeometryShader(NULL);
		SetPixelShader(CompileShader(ps_4_0, main_ps()));
	}
}