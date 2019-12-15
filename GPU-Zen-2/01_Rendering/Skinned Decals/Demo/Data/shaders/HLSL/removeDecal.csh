#include "globals.shi"

StructuredBuffer<float3> positionBuffer: register(t0, space0);
StructuredBuffer<float3> uvHandednessBuffer: register(t1, space0);
StructuredBuffer<uint> indexBuffer: register(t2, space0);
StructuredBuffer<DecalInfo> decalInfoBuffer: register(t3, space0);
#ifdef TYPED_UAV_LOADS
Texture2D<float4> decalLookupMaps[MAX_NUM_SUBMODELS]: register(t4, space0);
#else
Texture2D<uint> decalLookupMaps[MAX_NUM_SUBMODELS]: register(t4, space0);
#endif

RWStructuredBuffer<uint> decalValidityBuffer: register(u0, space0);

ConstantBuffer<DecalConstData> decalCB: register(b0, space0);

float3 GetBaryCentrics(in float3 rayOrigin, in float3 rayDir, in float3 pos0, in float3 pos1, in float3 pos2)
{
  float3 edge1 = pos1 - pos0;
  float3 edge2 = pos2 - pos0;

  float3 pvec = cross(rayDir, edge2);
  float det = dot(edge1, pvec);

  float3 tvec = rayOrigin - pos0;
  float u = dot(tvec, pvec);

  float3 qvec = cross(tvec, edge1);
  float v = dot(rayDir, qvec);

  float invDet = 1.0f / det;
  u *= invDet;
	v *= invDet;

  return float3(u, v, 1.0f - u - v);
}

[numthreads(1, 1, 1)]
void main()
{    
   uint decalHitMask = decalInfoBuffer[0].decalHitMask;
   if(decalHitMask != 0xffffffff)
   {
     uint triangleOffset = decalHitMask & 0xfffff;
     uint meshIndex = (decalHitMask >> 20u) & 0xf;
     uint iHitDistance = (decalHitMask >> 24u) & 0xff; 

     // calculate hit texCoords
     uint firstIndex = decalCB.subModelInfos[meshIndex].firstIndex;
     uint offset = triangleOffset * 3 + firstIndex;
     uint vertexIndex0 = indexBuffer[offset];
     uint vertexIndex1 = indexBuffer[offset + 1];
     uint vertexIndex2 = indexBuffer[offset + 2];
     float3 pos0 = positionBuffer[vertexIndex0];
     float3 pos1 = positionBuffer[vertexIndex1];
     float3 pos2 = positionBuffer[vertexIndex2];
     float2 texCoords0 = uvHandednessBuffer[vertexIndex0].xy;
     float2 texCoords1 = uvHandednessBuffer[vertexIndex1].xy;
     float2 texCoords2 = uvHandednessBuffer[vertexIndex2].xy;
     float3 baryCentrics = GetBaryCentrics(decalCB.rayOrigin.xyz, decalCB.rayDir.xyz, pos0, pos1, pos2);
     float2 texCoords = texCoords1 * baryCentrics.x + texCoords2 * baryCentrics.y + texCoords0 * baryCentrics.z;

     // calculate decal index
     float2 texSize;
     texSize.x = decalCB.subModelInfos[meshIndex].decalLookupMapWidth;
     texSize.y = decalCB.subModelInfos[meshIndex].decalLookupMapHeight;
     int2 iTexCoords = int2(texCoords * texSize - 0.5f);
#ifdef TYPED_UAV_LOADS
     uint decalIndex = uint(decalLookupMaps[meshIndex].Load(int3(iTexCoords, 0)).w * 255.0f + 0.0001f);
#else
     uint decalIndex = GetDecalIndex(decalLookupMaps[meshIndex].Load(int3(iTexCoords, 0)).x);
#endif

     // invalidate corresponding decal
     decalValidityBuffer[decalIndex] = 1;
   }
}
