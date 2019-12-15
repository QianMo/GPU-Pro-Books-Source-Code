cbuffer GlobalConst : register(b3)
{
  float4x4 g_ViewProjection;
  float4x4 g_ViewMat;
  float4x4 g_ProjMat;
  float4 g_CameraRight;
  float4 g_CameraUp;
  float4 g_CameraPos;
};
