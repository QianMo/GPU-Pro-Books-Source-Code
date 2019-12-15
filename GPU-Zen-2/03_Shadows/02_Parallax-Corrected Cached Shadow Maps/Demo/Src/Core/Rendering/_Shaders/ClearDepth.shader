float4 mainVS(float4 pos : POSITION) : SV_Position
{
  return float4(pos.xy, 0, 1);
}

float mainPS(float4 pos : SV_Position) : SV_Depth
{
  return 1;
}
