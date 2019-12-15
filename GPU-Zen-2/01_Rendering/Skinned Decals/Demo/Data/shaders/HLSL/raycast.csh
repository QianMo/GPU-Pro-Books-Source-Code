#include "globals.shi"

StructuredBuffer<float3> positionBuffer: register(t0, space0);
StructuredBuffer<uint> indexBuffer: register(t1, space0);

RWStructuredBuffer<DecalInfo> decalInfoBuffer: register(u0, space0);

ConstantBuffer<DecalMeshInfo> decalMeshInfoCB: register(b0, space0);
ConstantBuffer<DecalConstData> decalCB: register(b1, space0);

#define RAYCAST_THREAD_GROUP_SIZE 64 
#define EPSILON 0.0001f

float RayTriIntersect(in float3 rayOrigin, in float3 rayDir, in float3 pos0, in float3 pos1, in float3 pos2)
{
  float3 edge1 = pos1 - pos0;
  float3 edge2 = pos2 - pos0;

  float3 pvec = cross(rayDir, edge2);
  float det = dot(edge1, pvec);
  
  // back-face culling
	if(det < EPSILON)
		return -1.0f;

  float3 tvec = rayOrigin - pos0;
  float u = dot(tvec, pvec);
	if((u < 0.0f) || (u > det))
		return -1.0f;

  float3 qvec = cross(tvec, edge1);
  float v = dot(rayDir, qvec);
	if((v < 0.0f) || ((u+v) > det))
		return -1.0f;

	return (dot(edge2, qvec) / det);
}

[numthreads(RAYCAST_THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
   uint meshIndex = decalMeshInfoCB.decalMeshIndex;
   uint firstIndex = decalCB.subModelInfos[meshIndex].firstIndex;
   uint numTris = decalCB.subModelInfos[meshIndex].numTris;
   uint triangleOffset = dispatchThreadID.x;

   if(triangleOffset < numTris)
   {
     uint offset = triangleOffset * 3 + firstIndex;
     uint vertexIndex0 = indexBuffer[offset];
     uint vertexIndex1 = indexBuffer[offset + 1];
     uint vertexIndex2 = indexBuffer[offset + 2];
   
     float3 pos0 = positionBuffer[vertexIndex0];
     float3 pos1 = positionBuffer[vertexIndex1];
     float3 pos2 = positionBuffer[vertexIndex2];
    
     float hitDistance = RayTriIntersect(decalCB.rayOrigin.xyz, decalCB.rayDir.xyz, pos0, pos1, pos2);
     if(hitDistance >= 0.0f)
     {
       float hitDistanceN = (hitDistance - decalCB.hitDistances.x) * decalCB.hitDistances.z;
       uint iHitDistance = uint(hitDistanceN * 255.0f);

       uint decalHitMask = (iHitDistance << 24u) | (meshIndex << 20u) | triangleOffset;
       InterlockedMin(decalInfoBuffer[0].decalHitMask, decalHitMask);
     }
   }
}
