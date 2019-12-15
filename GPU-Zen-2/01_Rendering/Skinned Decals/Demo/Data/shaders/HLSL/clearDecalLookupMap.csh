#include "globals.shi"

#ifdef TYPED_UAV_LOADS
RWTexture2D<float4> decalLookupMaps[MAX_NUM_SUBMODELS]: register(u0, space0);
#else
RWTexture2D<uint> decalLookupMaps[MAX_NUM_SUBMODELS]: register(u0, space0);
#endif

ConstantBuffer<DecalMeshInfo> decalMeshInfoCB: register(b0, space0);
ConstantBuffer<DecalConstData> decalCB: register(b1, space0);

#define CLEAR_DECALLOOKUPMAP_THREAD_GROUP_SIZE 8

[numthreads(CLEAR_DECALLOOKUPMAP_THREAD_GROUP_SIZE, CLEAR_DECALLOOKUPMAP_THREAD_GROUP_SIZE, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
  uint meshIndex = decalMeshInfoCB.decalMeshIndex;
  uint width = decalCB.subModelInfos[meshIndex].decalLookupMapWidth;
  uint height = decalCB.subModelInfos[meshIndex].decalLookupMapHeight;

  if((dispatchThreadID.x < width) && (dispatchThreadID.y < height))
  {
#ifdef TYPED_UAV_LOADS
    decalLookupMaps[meshIndex][dispatchThreadID.xy] = 0;
#else
    decalLookupMaps[meshIndex][dispatchThreadID.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
#endif
  }
}
