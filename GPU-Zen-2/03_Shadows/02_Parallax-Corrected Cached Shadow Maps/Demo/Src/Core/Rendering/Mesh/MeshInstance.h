#ifndef __MESH_INSTANCE
#define __MESH_INSTANCE

#include "Scene/SceneObject.h"
#include "Mesh.h"
#include "../../Util/MemoryPool.h"

class MeshInstance : public SceneObject
{
  DECLARE_MEMORY_POOL();
public:
  MeshInstance(Mesh*, SceneObject* pParent = NULL, SceneQTreeNode* pQTreeRoot = NULL);
  void OptimizeAsStatic();

  virtual void OnTransformChanged() override;
  virtual Renderable* PrepareToRender() override;

protected:
  Mesh* m_Mesh;

  PushBuffer m_PrePassPB;
  Mesh::PrePassUpdateHandles m_PrePassUpdateHandles;
  PushBuffer& GetPrePassPB() { return m_PrePassPB; }

  PushBuffer m_ShadowMapPB;
  Mesh::ShadowMapUpdateHandles m_ShadowMapUpdateHandles;
  PushBuffer& GetShadowMapPB() { return m_ShadowMapPB; }

  PushBuffer m_CubeShadowMapPB;
  Mesh::CubeShadowMapUpdateHandles m_CubeShadowMapUpdateHandles;
  PushBuffer& GetCubeShadowMapPB() { return m_CubeShadowMapPB; }

  PushBuffer m_CubeShadowMapArrayPB;
  Mesh::CubeShadowMapArrayUpdateHandles m_CubeShadowMapArrayUpdateHandles;
  PushBuffer& GetCubeShadowMapArrayPB() { return m_CubeShadowMapArrayPB; }

  PushBuffer m_ParabolicShadowMapPB;
  Mesh::ParabolicShadowMapUpdateHandles m_ParabolicShadowMapUpdateHandles;
  PushBuffer& GetParabolicShadowMapPB() { return m_ParabolicShadowMapPB; }

  PushBuffer m_ASMLayerShadowMapPB;
  Mesh::ASMLayerShadowMapUpdateHandles m_ASMLayerShadowMapUpdateHandles;
  PushBuffer& GetASMLayerShadowMapPB() { return m_ASMLayerShadowMapPB; }

  friend class Mesh;
};

#endif //#ifndef __MESH_INSTANCE
