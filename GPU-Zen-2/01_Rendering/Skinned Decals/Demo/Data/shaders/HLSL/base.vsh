#include "globals.shi"

StructuredBuffer<float3> vertexPositions: register(t0, space0);
StructuredBuffer<float3> vertexNormals: register(t1, space0);
StructuredBuffer<float3> vertexTangents: register(t2, space0);
StructuredBuffer<float3> vertexUvHandedness: register(t3, space0);

ConstantBuffer<CameraConstData> cameraCB: register(b0, space0);

struct VS_Output
{
  float4 position: SV_POSITION;
  float2 texCoords: TEXCOORD;
  float3 normal: NORMAL;
  float3 tangent: TANGENT;
  float3 bitangent: BITANGENT;
  float3 positionWS: POSITION_WS;
};

VS_Output main(uint vertexID: SV_VertexID)
{
  VS_Output output;

  float3 position = vertexPositions[vertexID];
  float3 normal = vertexNormals[vertexID];
  float3 tangent = vertexTangents[vertexID];
  float3 uvHandedness = vertexUvHandedness[vertexID];

  output.position = mul(cameraCB.viewProjMatrix, float4(position, 1.0f));
  output.texCoords = uvHandedness.xy;
  output.normal = normal;
  output.tangent = tangent;
  output.bitangent = cross(normal, tangent) * uvHandedness.z;
  output.positionWS = position;

  return output;
}

