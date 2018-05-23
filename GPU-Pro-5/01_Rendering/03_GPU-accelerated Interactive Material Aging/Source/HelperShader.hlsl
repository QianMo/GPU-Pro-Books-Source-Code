#ifndef __HELPER_SHADERS__
#define __HELPER_SHADERS__

// This file is included in the atlas material

// some of the functions that are not directly involved in the material aging
// were removed from the effect to increate readablity


float4 g_DilatationMask = float4(1,1,1,1);

// Pixel Shader that simply renders a debug texture
float4 PS_DebugOutput( ATLAS_POSITION input) : SV_Target0
{
	//return float4(Tex_Unspecified.Sample( samplerLinear, input.TexCoord).xyz, 1);
	return float4(Tex_Unspecified.Load( int3(input.Position.xy, 0) ).xyz, 1);
}

// dilatation sample offsets
static const int3 neighbors[8] = { int3(-1, 0, 0), int3(1, 0, 0), int3(0, 1, 0), int3(0, -1, 0), int3(-1, 1, 0), int3(1, 1, 0), int3(-1, -1, 0), int3(1, -1, 0) };

// to reduce sampling error at texture seams
// we apply a dilatation filter on each texture of the atlas once during startup
float4 PS_Dilatation( ATLAS_POSITION input) : SV_Target0
{
	int3 currentPos = int3(input.Position.xy, 0);

	// value at the current texel
	float4 value = Tex_Unspecified.Load( currentPos ) * g_DilatationMask;

	// do not effect texels that contain information
	// we are just looking for texels at near seams that contain no information
	if(any(value)) // note that this can't be used for textures with "float4(0, 0, 0, 0)" as "desired feature".
		return value;

	int hits = 0;
	float4 accumValue = float4(0, 0, 0, 0);

	// sample the neightborhood 
	[unroll]
	for(int i=0; i<8; i++)
	{
		value = Tex_Unspecified.Load( currentPos + neighbors[i] ) * g_DilatationMask;
		if(any(value))
		{
			hits += 1;
			accumValue += value;
		}
	}

	// return the average of the found samples with data
	if(hits > 0)
		return accumValue / hits;

	// return "no data" if no neighbor contains information
	return float4(0,0,0,0);
}

#endif