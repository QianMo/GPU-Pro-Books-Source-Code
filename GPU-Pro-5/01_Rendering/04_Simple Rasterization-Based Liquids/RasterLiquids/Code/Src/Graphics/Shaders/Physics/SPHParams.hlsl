
cbuffer cbSPHParams : register( b3 )
{
	float4 GridSpacing;	
	float4 Gravity;
	
	float4 InitialDensity;

	float PressureScale;	
	float PIC_FLIP;
	float SurfaceDensity;
	
	float dt;
	
}