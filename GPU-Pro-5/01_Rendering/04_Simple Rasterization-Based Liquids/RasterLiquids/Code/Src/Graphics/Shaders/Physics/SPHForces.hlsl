
///<
float GetObstacleDensity(float3 _UVW)
{
	float3 sphereCenter = float3(0.7f,0.7f,0.2f);
	float sphereRadius=0.15f;
	
	if (length(_UVW-sphereCenter) <= sphereRadius)
		return (1200.0f/PressureScale)*(1.0f- length(_UVW-sphereCenter)/sphereRadius );
		
	return 0;
}

///<
float3 GetObstacleGradient(float3 _UVW)
{

float3 sphereCenter = float3(0.7f,0.7f,0.2f);
	float sphereRadius=0.15f;
	
	float l=length(_UVW-sphereCenter);
	if (l <= sphereRadius)
	{
		l=min(3.0f*l/sphereRadius, 1);
		l=l*l;

		//l=length(_UVW-sphereCenter)/sphereRadius;

		return 20.0f*(1.0f-l)*(-1.0f*l*_UVW);
	}
		
	return 0;

	/*
	float dR=GetObstacleDensity( _UVW+float3(GridSpacing.x,0,0) );
	float dL=GetObstacleDensity( _UVW-float3(GridSpacing.x,0,0) );	
	
	float dT=GetObstacleDensity( _UVW+float3(0,GridSpacing.y,0) );
	float dD=GetObstacleDensity( _UVW-float3(0,GridSpacing.y,0) );
	
	float dF=GetObstacleDensity( _UVW+float3(0,0,GridSpacing.z) );
	float dB=GetObstacleDensity( _UVW-float3(0,0,GridSpacing.z) );
	
	return float3(dR-dL,dT-dD,dF-dB); ///(2.0f*GridSpacing.xyz);
	*/
}

///< Surface Tension
float3 GetSurfaceTension(float3 _UVW)
{
	float d = GetDensity( _UVW );
	
	float dR=GetDensity( _UVW+float3(GridSpacing.x,0,0) );
	float dL=GetDensity( _UVW-float3(GridSpacing.x,0,0) );	
	
	float dT=GetDensity( _UVW+float3(0,GridSpacing.y,0) );
	float dD=GetDensity( _UVW-float3(0,GridSpacing.y,0) );
	
	float dF=GetDensity( _UVW+float3(0,0,GridSpacing.z) );
	float dB=GetDensity( _UVW-float3(0,0,GridSpacing.z) );
	
	float3 n = float3(dR-dL,dT-dD,dF-dB); 
	
	float mag = (dR+dL+dT+dD+dF+dB - 6.0*d);
	
	float lenN = length(n);
	if (lenN>0.005f )
		return -mag*n/lenN;
		
	return 0;
}

///<
float3 GetViscosity(float3 _UVW)
{

	float3 u = GetSplattedVelocity( _UVW, samPoint );
	
	float3 uR=GetSplattedVelocity( _UVW+float3(GridSpacing.x,0,0), samPoint );
	float3 uL=GetSplattedVelocity( _UVW-float3(GridSpacing.x,0,0), samPoint );	
	
	float3 uT=GetSplattedVelocity( _UVW+float3(0,GridSpacing.y,0), samPoint );
	float3 uD=GetSplattedVelocity( _UVW-float3(0,GridSpacing.y,0), samPoint );
	
	float3 uF=GetSplattedVelocity( _UVW+float3(0,0,GridSpacing.z), samPoint );
	float3 uB=GetSplattedVelocity( _UVW-float3(0,0,GridSpacing.z), samPoint );
	
	float3x3 Uplus	= {uR,uT,uF};	
	float3x3 Uminus	= {uL,uD,uB};	
		
	float3 Laplacian 			= mul(float3(1,1,1),Uplus) + mul(float3(1,1,1),Uminus) - 6.0f*u;
	
	return Laplacian/(GridSpacing.xyz*GridSpacing.xyz*4);

}