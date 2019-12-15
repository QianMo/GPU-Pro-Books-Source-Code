#ifndef __LIGHT_H
#define __LIGHT_H

#include <deque>
#include "Scene/SceneObject.h"
#include "LightBatch.h"
#include "_Shaders/HLSL2C.inc"
#include "_Shaders/Lighting.inc"
#include "../../Util/MemoryPool.h"

class DebugRenderer;

class Light : public SceneObject
{
public:
  Light(SceneObject* pParent, SceneQTreeNode* pQTreeRoot, LightBatch* pBatch) : m_Color(1.0f), m_Range(1.0f), m_UpdateTransform(false),
    m_NewPosition(Vec3::Zero()), SceneObject(Mat4x4::Identity(), pParent, pQTreeRoot), m_Batch(pBatch) { }
  virtual ~Light() { }

  finline const Vec4& GetColor() const { return m_Color; }
  finline float GetRange() const { return m_Range; }

  finline void SetRangeDeferred(float f) { m_Range = f; m_UpdateTransform = true; }
  finline void SetPositionDeferred(const Vec3& p) { m_NewPosition = p; m_UpdateTransform = true; }
  finline void SetColorDeferred(const Vec4& c) { m_Color = c; }

  virtual void Commit()
  {
    if(m_UpdateTransform)
    {
      QTreeNodeObject::Remove();
      if(m_Range!=0)
      {
        SetTransform(Mat4x4::ScalingTranslationD3D(Vec3(m_Range), m_NewPosition));
        m_UpdateTransform = false;
      }
    }
  }
  virtual void UpdateBSphereRadius() override
  {
    m_BSphereRadius = m_Range;
  }
  virtual Renderable* PrepareToRender() override
  {
    m_Batch->PrepareToRender(this);
    return NULL;
  }

protected:
  Vec4 m_Color;
  Vec3 m_NewPosition;
  LightBatch* m_Batch;
  float m_Range;
  bool m_UpdateTransform;
};

class PointLight : public Light
{
  DECLARE_MEMORY_POOL();
public:
  PointLight(SceneObject*, SceneQTreeNode*);

  void OnTransformChanged() override
  {
    __super::OnTransformChanged();
    UpdateShaderData();
  }
  void Commit() override
  {
    __super::Commit();
    UpdateShaderData();
  }

protected:
  PointLightShaderData m_ShaderData;

  finline void UpdateShaderData()
  {
    m_ShaderData.Position = GetPosition();
    m_ShaderData.Position.w = 1.0f/(m_Range*m_Range);
    m_ShaderData.Color = m_Color;
  }

  friend class PointLightBatch;
};

class CubeShadowMapPointLight : public PointLight
{
  DECLARE_MEMORY_POOL();
public:
  CubeShadowMapPointLight(unsigned, SceneObject*, SceneQTreeNode*, bool fastRender = false);
  ~CubeShadowMapPointLight();

  finline void Update(SceneRenderer* pRenderer, Scene* pScene, DeviceContext11& dc = Platform::GetImmediateContext())
  {
    static_cast<CubeShadowMapPointLightBatch*>(m_Batch)->Update(pRenderer, pScene, this, dc);
  }
  static void UpdateAll(SceneRenderer* pRenderer, Scene* pScene, DeviceContext11& dc = Platform::GetImmediateContext())
  {
    std::for_each(s_Lights.begin(), s_Lights.end(), [&] (CubeShadowMapPointLight* p) { p->Update(pRenderer, pScene, dc); });
  }

protected:
  Vec4i m_SMapRect;
  Vec4 m_SMapShaderData;
  int m_ArraySlice;
  bool m_FastRender;

  ptr_set_handle m_ListPos;
  ptr_set_handle& GetListPos() { return m_ListPos; }
  static ptr_set<CubeShadowMapPointLight, &CubeShadowMapPointLight::GetListPos> s_Lights;

  CubeShadowMapPointLight(SceneObject*, SceneQTreeNode*);
  const Mat4x4 GetViewMatrix(int) const;

  friend class CubeShadowMapPointLightBatch;
};

template<class T> class HemisphericalLight : public T
{
public:
  HemisphericalLight(SceneObject* pParent, SceneQTreeNode* pQTreeRoot) : T(pParent, pQTreeRoot), m_Normal(c_YAxis)
  {
    UpdateHLightShaderData();
  }
  finline void SetNormal(const Vec3& n)
  {
    m_Normal = n;
    UpdateHLightShaderData();
  }
  void OnTransformChanged() override
  {
    __super::OnTransformChanged();
    UpdateHLightShaderData();
  }

  finline const Vec3& GetNormal() const { return m_Normal; }
  finline const Vec4& GetHLightShaderData() const { return m_HLightShaderData; }

protected:
  Vec4 m_HLightShaderData;
  Vec3 m_Normal;

  finline void UpdateHLightShaderData()
  {
    m_HLightShaderData = m_Normal;
    m_HLightShaderData.w = -Vec3::Dot(m_Normal, GetPosition());
  }

  friend class HemisphericalPointLightBatch;
  friend class HemisphericalCubeShadowMapPointLightBatch;
};

class HemisphericalPointLight : public HemisphericalLight<PointLight>
{
  DECLARE_MEMORY_POOL();
public:
  HemisphericalPointLight(SceneObject*, SceneQTreeNode*);
};

class HemisphericalCubeShadowMapPointLight : public HemisphericalLight<CubeShadowMapPointLight>
{
  DECLARE_MEMORY_POOL();
public:
  HemisphericalCubeShadowMapPointLight(unsigned, SceneObject*, SceneQTreeNode*, bool fastRender = false);
};

class ReflectiveCubeShadowMapPointLight : public CubeShadowMapPointLight
{
  DECLARE_MEMORY_POOL();
public:
  ReflectiveCubeShadowMapPointLight(unsigned, SceneObject*, SceneQTreeNode*, unsigned NumVLights = 20, unsigned VLightSMapFaceSize = 64);
  ~ReflectiveCubeShadowMapPointLight();
  void DebugDrawVirtualLights(DebugRenderer&);

  finline float GetReflectivity() const { return m_Reflectivity; }
  finline void SetReflectivityDeferred(float f) { m_Reflectivity = f; }
  void Commit() override;

  static bool Process(SceneRenderer*, Scene*, DeviceContext11& dc = Platform::GetImmediateContext());

  finline void Update(SceneRenderer* pRenderer, Scene* pScene, DeviceContext11& dc = Platform::GetImmediateContext())
  {
    static_cast<ReflectiveCubeShadowMapPointLightBatch*>(m_Batch)->Update(pRenderer, pScene, this, dc);
  }
  static void UpdateAll(SceneRenderer* pRenderer, Scene* pScene, DeviceContext11& dc = Platform::GetImmediateContext())
  {
    std::for_each(s_Lights.begin(), s_Lights.end(), [&] (ReflectiveCubeShadowMapPointLight* p) { p->Update(pRenderer, pScene, dc); });
  }

protected:
  float m_Reflectivity;

  ptr_set_handle m_ListPos;
  ptr_set_handle& GetListPos() { return m_ListPos; }
  static ptr_set<ReflectiveCubeShadowMapPointLight, &ReflectiveCubeShadowMapPointLight::GetListPos> s_Lights;

  struct VirtualLightParam { Vec3 Position, Color, Normal; float Falloff, Area; };
  AlignedPODVector<VirtualLightParam> m_VLP;
  AlignedPODVector<VirtualLightParam> m_NewVLP;
  std::vector<HemisphericalPointLight*> m_VLights;
  std::vector<HemisphericalCubeShadowMapPointLight*> m_SMapVLights;

  typedef void (ReflectiveCubeShadowMapPointLight::*TaskProc)(unsigned, SceneRenderer*, Scene*, DeviceContext11&);
  struct Task { ReflectiveCubeShadowMapPointLight* pLight; TaskProc f; unsigned i; };
  static std::deque<Task> s_Tasks;

  void DeleteTasks();
  void FloodFill(unsigned, SceneRenderer*, Scene*, DeviceContext11&);
  void ReadClusterData(unsigned, SceneRenderer*, Scene*, DeviceContext11&);
  void UpdateShadowCastingVLight(unsigned, SceneRenderer*, Scene*, DeviceContext11&);
  void UpdateVLights(unsigned, SceneRenderer*, Scene*, DeviceContext11&);

  template<class T> void UpdateVirtualLightParam(T* pLight, const VirtualLightParam& vlp)
  {
    pLight->SetNormal(vlp.Normal);
    pLight->SetColorDeferred(GetColor()*vlp.Color*vlp.Area*m_Reflectivity);
    pLight->SetRangeDeferred(GetRange()*vlp.Falloff);
    pLight->SetPositionDeferred(vlp.Position);
    pLight->Commit();
  }

  friend class ReflectiveCubeShadowMapPointLightBatch;
};

#endif //#ifndef __LIGHT_H
