#include "globals.shi"

StructuredBuffer<WeightIndex> weightIndexBuffer: register(t0, space0);
StructuredBuffer<Weight> weightBuffer: register(t1, space0);
StructuredBuffer<Joint> jointBuffer: register(t2, space0);

RWStructuredBuffer<float3> positionBuffer: register(u0, space0);
RWStructuredBuffer<float3> normalBuffer: register(u1, space0);
RWStructuredBuffer<float3> tangentBuffer: register(u2, space0);

ConstantBuffer<ModelConstData> modelCB: register(b0, space0);

#define SKINNING_THREAD_GROUP_SIZE 64 

[numthreads(SKINNING_THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{       
  uint vertexIndex = dispatchThreadID.x;
  if(vertexIndex < modelCB.numVertices)
  {
    float3 vertexPosition = float3(0.0f, 0.0f, 0.0f);
    float3 vertexNormal = float3(0.0f, 0.0f, 0.0f);
    float3 vertexTangent = float3(0.0f, 0.0f, 0.0f);

    WeightIndex weightIndex = weightIndexBuffer[vertexIndex];
    for(uint i=0; i<weightIndex.numIndices; i++)
    {
      Weight weight = weightBuffer[weightIndex.firstIndex + i];
      Joint joint = jointBuffer[weight.jointIndex];
      vertexPosition += (joint.translation + MulQuatVec(joint.rotation, weight.position)) * weight.weight;
      vertexNormal += MulQuatVec(joint.rotation, weight.normal) * weight.weight;
      vertexTangent += MulQuatVec(joint.rotation, weight.tangent) * weight.weight;
    }
    vertexNormal = normalize(vertexNormal);
    vertexTangent = normalize(vertexTangent);

    positionBuffer[vertexIndex] = mul(modelCB.transformMatrix, float4(vertexPosition, 1.0f)).xyz;
    normalBuffer[vertexIndex] = mul((float3x3)modelCB.transformMatrix, vertexNormal);
    tangentBuffer[vertexIndex] = mul((float3x3)modelCB.transformMatrix, vertexTangent);
  }
}
