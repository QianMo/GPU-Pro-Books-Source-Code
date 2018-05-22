#include "globals.shi"

struct GS_OUTPUT
{
  float4 position: SV_POSITION;
  float2 texCoords: TEXCOORD;
};

struct FS_OUTPUT
{
  float4 fragColor: SV_TARGET;
};

#define SKY_COLOR float4(0.26f,0.49f,0.92f,1.0f)

FS_OUTPUT main(GS_OUTPUT input) 
{
  FS_OUTPUT output;
	output.fragColor = SKY_COLOR;
	return output;
}