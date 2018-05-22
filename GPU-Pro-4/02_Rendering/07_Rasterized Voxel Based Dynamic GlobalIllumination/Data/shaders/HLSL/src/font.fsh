#include "globals.shi"

Texture2D colorMap: register(COLOR_TEX_BP);
SamplerState colorMapSampler: register(COLOR_SAM_BP);

struct GS_OUTPUT
{
  float4 position: SV_POSITION;
  float2 texCoords: TEXCOORD;
	float4 color: COLOR;
};

struct FS_OUTPUT
{
  float4 fragColor: SV_TARGET;
};

FS_OUTPUT main(GS_OUTPUT input) 
{
  FS_OUTPUT output;
	float4 base = colorMap.Sample(colorMapSampler,input.texCoords);
	output.fragColor = base*input.color;
	return output;
}