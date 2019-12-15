float4 mainVS(float4 pos : POSITION) : SV_Position
{
  return float4(pos.xy, 0, 1);
}

cbuffer Constants : register(b0)
{
  uint2 g_MapExt;
  float g_MapWidth;
};

float mainPS(float4 pos : SV_Position) : SV_Target
{
  uint c_Mask = 63;
  uint2 i = uint2(pos.xy);
  uint2 t = i & ~c_Mask;
  float2 f = float2(t | (min(c_Mask, g_MapExt - t) - (i & c_Mask)));
  return f.y*g_MapWidth + f.x;
}
