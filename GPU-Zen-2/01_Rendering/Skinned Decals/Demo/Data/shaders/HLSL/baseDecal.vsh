#include "globals.shi"

StructuredBuffer<float3> vertexPositions: register(t0, space0);
StructuredBuffer<float3> vertexNormals: register(t1, space0);
StructuredBuffer<float3> vertexUvHandedness: register(t2, space0);
StructuredBuffer<DecalInfo> decalInfoBuffer: register(t3, space0);

struct VS_Output
{
	float4 position: SV_POSITION;
  float2 decalTC: DECAL_TEXCOORDS;
	float3 normal: NORMAL;
  float4 clipDistances: SV_ClipDistance;
};

VS_Output main(uint vertexID: SV_VertexID)
{
  VS_Output output;

  float3 position = vertexPositions[vertexID];
  float3 normal = vertexNormals[vertexID];
  float2 texCoords = vertexUvHandedness[vertexID].xy;

  // transform position from object to decal space
  float3 decalTC = mul(decalInfoBuffer[0].decalMatrix, float4(position, 1.0f)).xyz;

  output.decalTC = decalTC.xy;

  // Write rasterized fragments to output texture corresponding to the input UVs and use z-component of decal UV 
  // to clip decal in z-direction.
  output.position = float4(float2(texCoords.x, 1.0f - texCoords.y) * 2.0f - 1.0f, decalTC.z, 1.0f);
  output.position.xy = output.position.xy * decalInfoBuffer[0].decalScale + decalInfoBuffer[0].decalBias;

  // clip decal in x- and y-direction
  output.clipDistances = float4(decalTC.x, 1.0f - decalTC.x, decalTC.y, 1.0f - decalTC.y);

  output.normal = normal;

  return output;
}

