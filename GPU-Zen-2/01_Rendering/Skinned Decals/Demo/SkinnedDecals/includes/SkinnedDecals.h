#ifndef SKINNED_DECALS_H
#define SKINNED_DECALS_H

#include <DX12_ResourceDescTable.h>

#define CLEAR_DECALLOOKUPMAP_THREAD_GROUP_SIZE 8
#define RAYCAST_THREAD_GROUP_SIZE 64
#define MAX_NUM_SUBMODELS 16
#define MAX_NUM_SKINNED_DECALS 256
#define MAX_NUM_DECAL_MATERIALS 8
#define MAX_NUM_DECAL_TEXTURES (MAX_NUM_DECAL_MATERIALS * 3)

class DX12_PipelineState;
class DX12_Buffer;
class DX12_RenderTarget;
class DX12_CmdSignature;
class ViewportSet;
class ScissorRectSet;
class DemoModel;

struct AddDecalInfo
{
  Vector3 rayOrigin;
  Vector3 rayDir;
  Vector3 decalTangent;
  Vector3 decalSize;
  float minHitDistance;
  float maxHitDistance;
  UINT decalMaterialIndex;
};

struct RemoveDecalInfo
{
  Vector3 rayOrigin;
  Vector3 rayDir;
  float minHitDistance;
  float maxHitDistance;
};

// SkinnedDecals
//
// Adds dynamically decals to a skinned model. Since skinning of models is done on the GPU via compute shader, 
// the entire process from finding the closest intersection between the ray that places the decal and the model,
// to adding the decal to the model, is done on the GPU. After finding the submodel/ position/ normal of the 
// closest intersection, the corresponding information is written into a GPU buffer, that is used to indirectly 
// render the decal into the decal lookup texture of the submodel. This texture is used in the base pass to apply 
// the decal material to the base material of the model. A decal validity buffer ensures on the GPU, that decals
// don't overlap. The implementation supports adding, removing individual decals and clearing all decals.
class SkinnedDecals
{
public:
  friend class DemoModel;

  struct SubModelInfo
  {
    UINT firstIndex;
    UINT numTris;
    float decalLookupMapWidth;
    float decalLookupMapHeight;
  };

  struct DecalConstData
  {
    SubModelInfo subModelInfos[MAX_NUM_SUBMODELS];
    Vector4 rayOrigin;
    Vector4 rayDir;
    Vector4 decalTangent;
    Vector4 hitDistances; // x = minHitDistance, y = maxHitDistance - minHitDistance, z = 1.0f / y
    float decalLookupRtWidth;
    float decalLookupRtHeight;
    float decalSizeX;
    float decalSizeY;
    float decalSizeZ;
    UINT decalIndex;
    UINT decalMaterialIndex;
  };

  struct DrawIndirectCmd
  {
    UINT indexCountPerInstance;
    UINT instanceCount;
    UINT startIndexLocation;
    int baseVertexLocation;
    UINT startInstanceLocation;
  };

  struct DecalInfo
  {
    DrawIndirectCmd drawCmd;
    UINT decalHitMask; // bit 0-19: triangleIndex, bit 20-23: subModelIndex, bit 24-31: hitDistance
    UINT decalIndex;
    Vector2 decalScale;
    Vector2 decalBias;
    Vector3 decalNormal;
    Matrix4 decalMatrix;
    UINT decalMaterialIndices[MAX_NUM_SKINNED_DECALS];
  };

  SkinnedDecals():
    parentModel(nullptr),
    numDecalMaterials(0),
    clearSkinnedDecals(false),
    addSkinnedDecal(false),
    removeSkinnedDecal(false),
    numDecalLookupTextures(0),
    decalLookupMapRT(nullptr),
    decalLookupMapVPS(nullptr),
    decalLookupMapSRS(nullptr),
    decalValiditySB(nullptr),
    decalInfoSB(nullptr),
    decalCB(nullptr),
    clearDecalLookupMapPS(nullptr),
    clearDecalValidityBufferPS(nullptr),
    clearDecalInfoPS(nullptr),
    raycastPS(nullptr),
    calcDecalInfoPS(nullptr),
    renderDecalPS(nullptr),
    renderDecalCS(nullptr),
    removeDecalPS(nullptr)
  {
    memset(decalLookupTextures, 0, sizeof(DX12_Texture*) * MAX_NUM_SUBMODELS);
  }

  bool Init(const DemoModel *parentModel, const char** materialNames, UINT numMaterials);

	void Render();

  void ClearSkinnedDecals()
  {
    // reset decalIndex (0 reserved for invalid decal index)
    decalConstData.decalIndex = 0;

    clearSkinnedDecals = true;
  }

  void AddSkinnedDecal(const AddDecalInfo &decalInfo);

  void RemoveSkinnedDecal(const RemoveDecalInfo &decalInfo);

  bool RequiresSkinningData() const
  {
    return (addSkinnedDecal || removeSkinnedDecal);
  }

private:	 
  void UpdateBuffers();

  void PerformClearSkinnedDecals();

  void PerformAddSkinnedDecal();

  void PerformRemoveSkinnedDecal();

  const DemoModel *parentModel;
  DecalConstData decalConstData;
  UINT numDecalMaterials;
  bool clearSkinnedDecals;
  bool addSkinnedDecal;
  bool removeSkinnedDecal;

  DX12_Texture *decalLookupTextures[MAX_NUM_SUBMODELS];
  UINT numDecalLookupTextures;
  DX12_RenderTarget *decalLookupMapRT;
  ViewportSet *decalLookupMapVPS;
  ScissorRectSet *decalLookupMapSRS;
  DX12_Buffer *decalValiditySB;
  DX12_Buffer *decalInfoSB;
  DX12_Buffer *decalCB;

  DX12_PipelineState *clearDecalLookupMapPS;
  DX12_PipelineState *clearDecalValidityBufferPS;
  DX12_PipelineState *clearDecalInfoPS;
  DX12_PipelineState *raycastPS;
  DX12_PipelineState *calcDecalInfoPS;
  DX12_PipelineState *renderDecalPS;
  DX12_CmdSignature *renderDecalCS;
  DX12_PipelineState *removeDecalPS;

  DX12_ResourceDescTable clearDecalsDT;
  DX12_ResourceDescTable clearDecalInfoDT;
  DX12_ResourceDescTable calcDecalInfoDT;
  DX12_ResourceDescTable renderDecalDT;
  DX12_ResourceDescTable removeDecalDT;
  DX12_ResourceDescTable decalBuffersDT;
  DX12_ResourceDescTable decalMaterialsDT;

};

#endif