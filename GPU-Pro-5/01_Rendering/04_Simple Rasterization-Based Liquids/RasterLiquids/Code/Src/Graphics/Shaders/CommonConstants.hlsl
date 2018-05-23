

cbuffer cbCameraMatrices : register( b0 )
{
	float4 ScreenDimensions;
	float4 CamPos;
	matrix View;
	matrix Proj;   
	matrix ViewInv;

	matrix LightView;
}

cbuffer cbObjectMatrix : register( b1 )
{
	matrix World;	
}

SamplerState	samPoint	: register( s0 );
SamplerState	samLinear	: register( s1 );