#ifndef __SCENE_RENDERER
#define __SCENE_RENDERER

#include "Scene/Traversal.h"
#include "Scene/Renderable.h"

class Camera;
class Scene;
struct ID3D11Buffer;

class SceneRenderer
{
public:
  typedef std::vector<Renderable*> RenderList;

  SceneRenderer()
  {
    m_RenderList.reserve(2048);
  }
  void DrawPrePass(Scene* pScene, Camera* pCamera, float contribCullThreshold = 0.0005f)
  {
    BuildRenderList(pScene, pCamera, contribCullThreshold);
    DrawPrePass(m_RenderList, pCamera);
  }
  void DrawShadowMap(Scene* pScene, Camera* pCamera, float contribCullThreshold = 0.0f)
  {
    BuildRenderList(pScene, pCamera, contribCullThreshold);
    DrawShadowMap(m_RenderList, pCamera);
  }
  void DrawCubeShadowMap(Scene* pScene, Camera* pCamera, float contribCullThreshold = 0.0005f)
  {
    BuildRenderList(pScene, pCamera, contribCullThreshold);
    DrawCubeShadowMap(m_RenderList, pCamera);
  }
  void DrawCubeShadowMapArray(Scene* pScene, Camera* pCamera, float contribCullThreshold = 0.0005f)
  {
    BuildRenderList(pScene, pCamera, contribCullThreshold);
    DrawCubeShadowMapArray(m_RenderList, pCamera);
  }
  void DrawParabolicShadowMap(Scene* pScene, Camera* pCamera, float contribCullThreshold = 0.0005f)
  {
    BuildRenderList(pScene, pCamera, contribCullThreshold);
    DrawParabolicShadowMap(m_RenderList, pCamera);
  }
  void DrawASMLayerShadowMap(Scene* pScene, Camera* pCamera, float contribCullThreshold = 0.0f)
  {
    BuildRenderList(pScene, pCamera, contribCullThreshold);
    DrawASMLayerShadowMap(m_RenderList, pCamera);
  }

  virtual void DrawPrePass(const RenderList&, Camera*);
  virtual void DrawShadowMap(const RenderList&, Camera*);
  virtual void DrawCubeShadowMap(const RenderList&, Camera*);
  virtual void DrawCubeShadowMapArray(const RenderList&, Camera*);
  virtual void DrawParabolicShadowMap(const RenderList&, Camera*);
  virtual void DrawASMLayerShadowMap(const RenderList&, Camera*);

protected:
  SceneQTreeTraversal m_TravelAgent;
  RenderList m_RenderList;

  finline void BuildRenderListCallback(SceneObject* pObj)
  {
    Renderable* pRenderable = pObj->PrepareToRender();
    if(pRenderable!=NULL)
      m_RenderList.push_back(pRenderable);
  }
  template<void (Renderable::*Draw)(DeviceContext11&)> void DrawRenderList(const RenderList& renderList, Camera* pCamera)
  {
    DeviceContext11& dc = Platform::GetImmediateContext();
    dc.PushRC();

    ID3D11Buffer* pBuffer = GetGlobalConstants(pCamera, dc);
    dc.VSSetConstantBuffer(3, pBuffer);
    dc.GSSetConstantBuffer(3, pBuffer);
    dc.PSSetConstantBuffer(3, pBuffer);
    dc.HSSetConstantBuffer(3, pBuffer);
    dc.DSSetConstantBuffer(3, pBuffer);

    for(auto it=renderList.cbegin(); it!=renderList.cend(); ++it)
      ((*it)->*Draw)(dc);

    dc.GetConstantBuffers().Free(pBuffer);
    dc.PopRC();
  }

  void BuildRenderList(Scene*, Camera*, float);
  ID3D11Buffer* GetGlobalConstants(Camera*, DeviceContext11&);
};

class TBBObserver;
class TBBInit;

class ParallelSceneRenderer : public SceneRenderer
{
public:
  ParallelSceneRenderer() : m_TBBInit(NULL), m_TBBObserver(NULL) { }
  void Init();
  void Shutdown();

  virtual void DrawPrePass(const RenderList&, Camera*) override;
  virtual void DrawShadowMap(const RenderList&, Camera*) override;
  virtual void DrawCubeShadowMap(const RenderList&, Camera*) override;
  virtual void DrawCubeShadowMapArray(const RenderList&, Camera*) override;
  virtual void DrawParabolicShadowMap(const RenderList&, Camera*) override;
  virtual void DrawASMLayerShadowMap(const RenderList&, Camera*) override;

protected:
  TBBInit* m_TBBInit;
  TBBObserver* m_TBBObserver;
};

#endif //#ifndef __SCENE_RENDERER
