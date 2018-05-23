

///<
float1 PS_ComputeDivergence(GS_OUTPUT _input) : SV_Target
{	
	const float3 UVW = _input.UV;

	float dR=txUp.Sample(samPoint, UVW+float3(GridSpacing.x,0,0) ).x;
	float dL=txUp.Sample(samPoint, UVW-float3(GridSpacing.x,0,0) ).x;	
	
	float dT=txUp.Sample(samPoint, UVW+float3(0,GridSpacing.y,0) ).y;
	float dD=txUp.Sample(samPoint, UVW-float3(0,GridSpacing.y,0) ).y;
	
	float dF=txUp.Sample(samPoint, UVW+float3(0,0,GridSpacing.z) ).z;
	float dB=txUp.Sample(samPoint, UVW-float3(0,0,GridSpacing.z) ).z;
	
	float div=0.0f;
	if (!IsBoundary(UVW))
		div = (dR-dL + dT-dD + dF-dB)*0.5f;
	
	return div;

}

///<
float GetPressure(float3 _UVW)
{
	float d = txUp.Sample(samPoint,_UVW).w;
	
	if (d>4.0f)
		return txP.Sample(samPoint,_UVW).x;
	else 
		return 0;
}

///< Compute Jacobi Iterations
float1 PS_Jacobi(GS_OUTPUT _input) : SV_Target
{
	const float c = -6.0f;
	const float a = 1.0f;
	
	const float3 UVW = _input.UV;

	float p = txUp.Sample(samPoint,UVW).w;
	if (p<4.0f)
		return 0;
	
	float PC = GetPressure(UVW);
	
	float PL = GetPressure(UVW-float3(GridSpacing.x,0,0));
	float PR = GetPressure(UVW+float3(GridSpacing.x,0,0));
	
	float PT = GetPressure(UVW+float3(0,GridSpacing.y,0));
	float PD = GetPressure(UVW-float3(0,GridSpacing.y,0));
	
	float PF = GetPressure(UVW+float3(0,0,GridSpacing.z));
	float PB = GetPressure(UVW-float3(0,0,GridSpacing.z));
	
	if (IsBoundary(UVW-float3(GridSpacing.x,0,0)))
		PL=PC;	
	if (IsBoundary(UVW+float3(GridSpacing.x,0,0)))
		PR=PC;
	if (IsBoundary(UVW+float3(0,GridSpacing.y,0)))
		PT=PC;
	if (IsBoundary(UVW-float3(0,GridSpacing.y,0)))
		PD=PC;
	if (IsBoundary(UVW+float3(0,0,GridSpacing.z)))
		PF=PC;
	if (IsBoundary(UVW-float3(0,0,GridSpacing.z)))
		PB=PC;

	float Div = txDiv.Sample(samPoint,UVW);
	float Pressure = (Div-a*(PD+PT+PL+PR+PF+PB))/c;
	
	return Pressure;
}



///<
float4 PS_AddPressureGradient(GS_OUTPUT PSInput) : SV_Target
{
	const float3 UVW 			= PSInput.UV;
	
	float4 samp = txUp.Sample(samPoint,UVW).xyzw;

	if (!IsBoundary(UVW))
	{
		float PC		= GetPressure(UVW);
	
		float PR 		= GetPressure(UVW+float3(GridSpacing.x,0,0));
		float PL	 	= GetPressure(UVW-float3(GridSpacing.x,0,0));
		
		float PT		= GetPressure(UVW+float3(0,GridSpacing.y,0));
		float PD		= GetPressure(UVW-float3(0,GridSpacing.y,0));
		
		float PF		= GetPressure(UVW+float3(0,0,GridSpacing.z));
		float PB		= GetPressure(UVW-float3(0,0,GridSpacing.z));
			
		if (IsBoundary(UVW-float3(GridSpacing.x,0,0)))
			PL=PC;	
		if (IsBoundary(UVW+float3(GridSpacing.x,0,0)))
			PR=PC;
		if (IsBoundary(UVW+float3(0,GridSpacing.y,0)))
			PT=PC;
		if (IsBoundary(UVW-float3(0,GridSpacing.y,0)))
			PD=PC;
		if (IsBoundary(UVW+float3(0,0,GridSpacing.z)))
			PF=PC;
		if (IsBoundary(UVW-float3(0,0,GridSpacing.z)))
			PB=PC;
			
		float3 GradP = 0.5f*float3(PR-PL,PT-PD,PF-PB);
		
		samp.xyz = samp.xyz - GradP;
	}
	
	return samp;
		
}