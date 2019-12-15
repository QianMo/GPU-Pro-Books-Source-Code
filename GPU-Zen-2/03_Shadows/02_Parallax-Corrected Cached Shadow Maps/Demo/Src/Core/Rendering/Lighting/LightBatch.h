#ifndef __LIGHT_BATCH
#define __LIGHT_BATCH

#include "../../Util/AlignedVector.h"
#include "../../Util/ptr_set.h"
#include "TextureLoader/Texture11.h"

class Light;
class LightBuffer;
class Camera;
class Scene;
class SceneRenderer;
class CubeShadowMapPointLight;
class ReflectiveCubeShadowMapPointLight;
class HemisphericalCubeShadowMapPointLight;
class StructuredBuffer;
struct LightingShaderFlags;
struct LightingQuadShaderFlags;
class ShaderObject;

class LightBatch
{
public:
  LightBatch() { m_ToRender.reserve(1024); s_Batches.insert(this); }
  virtual ~LightBatch() { s_Batches.remove(this); }

  finline void PrepareToRender(Light* pLight)
  {
    m_ToRender.push_back(pLight);
  }
  static void ClearVisibilityInfo()
  {
    std::for_each(s_Batches.begin(), s_Batches.end(), [] (LightBatch* b) { b->m_ToRender.clear(); });
  }

  virtual void Draw(LightBuffer*, const StructuredBuffer*, RenderTarget2D*, RenderTarget2D*, RenderTarget2D*, Camera*, bool, DeviceContext11&) = 0;

protected:
  std::vector<Light*> m_ToRender;

  ptr_set_handle m_ListPos;
  ptr_set_handle& GetListPos() { return m_ListPos; }
  static ptr_set<LightBatch, &LightBatch::GetListPos> s_Batches;

  static const ShaderObject& GetShader(const LightingShaderFlags&);
  static const ShaderObject& GetShader(const LightingQuadShaderFlags&);

  friend class LightBuffer;
  friend class SharedResources;
};

class PointLightBatch : public LightBatch
{
public:
  virtual void Draw(LightBuffer*, const StructuredBuffer*, RenderTarget2D*, RenderTarget2D*, RenderTarget2D*, Camera*, bool, DeviceContext11&) override;

protected:
  void Draw(const LightingShaderFlags&, LightBuffer*, const StructuredBuffer*, RenderTarget2D*, RenderTarget2D*, RenderTarget2D*, Camera*, DeviceContext11&);
};

class CubeShadowMapPointLightBatch : public LightBatch
{
public:
  CubeShadowMapPointLightBatch(unsigned);
  ~CubeShadowMapPointLightBatch();

  static void Allocate(CubeShadowMapPointLight*, unsigned);
  void Add(CubeShadowMapPointLight*);
  void Free(CubeShadowMapPointLight*);
  void Update(SceneRenderer*, Scene*, CubeShadowMapPointLight*, DeviceContext11&);

  virtual void Draw(LightBuffer*, const StructuredBuffer*, RenderTarget2D*, RenderTarget2D*, RenderTarget2D*, Camera*, bool, DeviceContext11&) override;

protected:
  RenderTarget2D m_DepthAtlas;
  AlignedPODVector<Vec4i> m_FreeRect;
  unsigned m_NAllocated, m_FaceSize;
  unsigned m_RectWidth, m_RectHeight;
  unsigned m_PCFKernelSize;

  RenderTarget2D m_CBMArray;
  std::vector<int> m_FreeSlice;

  ptr_set_handle m_ListPos;
  ptr_set_handle& GetListPos() { return m_ListPos; }
  static ptr_set<CubeShadowMapPointLightBatch, &CubeShadowMapPointLightBatch::GetListPos> s_Batches;

  bool OnPlatformInit();
  void OnPlatformShutdown();
  const Mat4x4 GetProjectionMatrix(float) const;
  void Draw(const LightingShaderFlags&, LightBuffer*, const StructuredBuffer*, RenderTarget2D*, RenderTarget2D*, RenderTarget2D*, Camera*, DeviceContext11&);

  friend class ReflectiveCubeShadowMapPointLightBatch;
  friend class HemisphericalCubeShadowMapPointLightBatch;
};

class ReflectiveCubeShadowMapPointLightBatch : protected CubeShadowMapPointLightBatch
{
public:
  ReflectiveCubeShadowMapPointLightBatch(unsigned);
  ~ReflectiveCubeShadowMapPointLightBatch();

  void Update(SceneRenderer*, Scene*, ReflectiveCubeShadowMapPointLight*, DeviceContext11&);
  void FloodFill(int, ReflectiveCubeShadowMapPointLight*, DeviceContext11&);
  void ReadClusterData(int, ReflectiveCubeShadowMapPointLight*, DeviceContext11&);

  static void Allocate(ReflectiveCubeShadowMapPointLight*, unsigned);

protected:
  static const unsigned c_NReadbackTextures = 6;

  RenderTarget2D m_AlbedoRT[6];
  RenderTarget2D m_GeomNormalRT[6];
  RenderTarget2D m_DepthRT[6];
  RenderTarget2D m_PositionBuffer;
  RenderTarget2D m_FloodMap;
  RenderTarget2D m_IDs[2];
  RenderTarget2D m_IDReadback[c_NReadbackTextures];
  RenderTarget2D m_ColorReadback[c_NReadbackTextures];
  RenderTarget2D m_NormalReadback[c_NReadbackTextures];
  RenderTarget2D m_PositionReadback[c_NReadbackTextures];

  bool OnPlatformInit();
  void OnPlatformShutdown();
  void CopyFaceDepth(RenderTarget2D&, const Vec4i&, int, DeviceContext11&);

  ptr_set_handle m_ListPos;
  ptr_set_handle& GetListPos() { return m_ListPos; }
  static ptr_set<ReflectiveCubeShadowMapPointLightBatch, &ReflectiveCubeShadowMapPointLightBatch::GetListPos> s_Batches;
};

class HemisphericalPointLightBatch : public PointLightBatch
{
public:
  virtual void Draw(LightBuffer*, const StructuredBuffer*, RenderTarget2D*, RenderTarget2D*, RenderTarget2D*, Camera*, bool, DeviceContext11&) override;
};

class HemisphericalCubeShadowMapPointLightBatch : public CubeShadowMapPointLightBatch
{
public:
  HemisphericalCubeShadowMapPointLightBatch(unsigned faceSize) : CubeShadowMapPointLightBatch(faceSize) { }

  static void Allocate(HemisphericalCubeShadowMapPointLight*, unsigned);

  virtual void Draw(LightBuffer*, const StructuredBuffer*, RenderTarget2D*, RenderTarget2D*, RenderTarget2D*, Camera*, bool, DeviceContext11&) override;
};

#endif //#ifndef __LIGHT_BATCH
