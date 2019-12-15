struct InputVS
{
  float4 Position : POSITION;
  float4 Color    : COLOR0;
};

struct OutputVS
{
  float4 Position : SV_Position;
  float4 Color    : COLOR0;
};

OutputVS mainVS(InputVS In)
{
  OutputVS Out;
  Out.Position = In.Position;
  Out.Color = In.Color;
  return Out;
}

float4 mainPS(OutputVS In) : SV_Target
{
  return In.Color;
}
