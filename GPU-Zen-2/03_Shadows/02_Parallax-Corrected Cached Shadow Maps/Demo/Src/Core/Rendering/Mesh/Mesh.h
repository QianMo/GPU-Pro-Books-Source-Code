#ifndef __MESH_H
#define __MESH_H

#include "PushBuffer/PushBuffer.h"
#include "Scene/Renderable.h"

class MaterialDesc;
class IABuffer;
class TextureLoader;
class Texture2D;
class MeshInstance;

class Mesh : public MathLibObject, public Renderable
{
public:
  Mesh();
  ~Mesh();
  bool LoadXML(const char*, TextureLoader&);

  virtual void DrawPrePass(DeviceContext11&);
  virtual void DrawShadowMap(DeviceContext11&);
  virtual void DrawCubeShadowMap(DeviceContext11&);
  virtual void DrawCubeShadowMapArray(DeviceContext11&);
  virtual void DrawParabolicShadowMap(DeviceContext11&);
  virtual void DrawASMLayerShadowMap(DeviceContext11&);

protected:
  static const unsigned c_NInstancesMax = 16;

  PushBuffer m_PrePassPB;
  struct PrePassUpdateHandles { PushBufferConstantHandle hTransform; } m_PrePassUpdateHandles;
  PushBuffer m_ShadowMapPB;
  struct ShadowMapUpdateHandles { PushBufferConstantHandle hTransform; } m_ShadowMapUpdateHandles;
  PushBuffer m_CubeShadowMapPB;
  struct CubeShadowMapUpdateHandles { PushBufferConstantHandle hTransform; } m_CubeShadowMapUpdateHandles;
  PushBuffer m_CubeShadowMapArrayPB;
  struct CubeShadowMapArrayUpdateHandles { PushBufferConstantHandle hTransform; } m_CubeShadowMapArrayUpdateHandles;
  PushBuffer m_ParabolicShadowMapPB;
  struct ParabolicShadowMapUpdateHandles { PushBufferConstantHandle hTransform; } m_ParabolicShadowMapUpdateHandles;
  PushBuffer m_ASMLayerShadowMapPB;
  struct ASMLayerShadowMapUpdateHandles { PushBufferConstantHandle hTransform; } m_ASMLayerShadowMapUpdateHandles;

  std::string m_Info;
  std::vector<MeshInstance*> m_ToRender;
  std::vector<Texture2D*> m_Textures;
  std::vector<IABuffer*> m_Buffers;
  Mat4x4 m_AABB;

  bool AddMaterial(const MaterialDesc&, TextureLoader&);
  void DrawInstanced(PushBuffer&, PushBufferConstantHandle&, DeviceContext11&);

  template<PushBuffer& (MeshInstance::*GetPB)()>
    finline void Draw(PushBuffer& instPB, PushBufferConstantHandle& hTransform, DeviceContext11& dc)
  {
    if(m_ToRender.size()<3)
    {
      for(auto it=m_ToRender.begin(); it!=m_ToRender.end(); ++it)
        ((*it)->*GetPB)().Execute(dc);
    }
    else
    {
      DrawInstanced(instPB, hTransform, dc);
    }
    m_ToRender.clear();
  }

  friend class MeshInstance;
};

#endif //#ifndef __MESH_H
