struct Input
{
  float4 Position : POSITION0;
  float4 Color    : COLOR0;
};

struct Output
{
  float4 Position : POSITION0;
  float4 Color    : COLOR0;
};

Output main(Input In)
{
  Output Out;
  Out.Position = In.Position;
  Out.Color = In.Color;
  return Out;
}
