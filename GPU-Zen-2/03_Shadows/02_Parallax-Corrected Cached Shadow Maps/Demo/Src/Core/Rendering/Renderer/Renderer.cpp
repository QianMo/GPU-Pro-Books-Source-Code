#include "PreCompile.h"
#include "Renderer.h"
#include "Scene/Camera.h"
#include "Scene/Scene.h"
#include "Lighting/LightBatch.h"
#include "Platform11/DeferredContext11.h"
#include "_Shaders/HLSL2C.inc"
#include "_Shaders/GlobalConst.inc"
#include "../../Util/Log.h"

void SceneRenderer::BuildRenderList(Scene* pScene, Camera* pCamera, float contribCullThreshold)
{
  LightBatch::ClearVisibilityInfo();

  m_TravelAgent.ProcessVisiblePrepare(pCamera->GetViewProjection(), pCamera->GetPosition(), pScene->GenerateTimeStamp());
  m_TravelAgent.SetContributionCullingScreenAreaThreshold(pCamera->GetProjection(), contribCullThreshold);

  m_RenderList.clear();
  m_TravelAgent.ProcessVisible<SceneRenderer, &SceneRenderer::BuildRenderListCallback>(this, pScene->GetQTreeRoot());
}

void SceneRenderer::DrawPrePass(const RenderList& renderList, Camera* pCamera)
{
  DrawRenderList<&Renderable::DrawPrePass>(renderList, pCamera);
}

void SceneRenderer::DrawShadowMap(const RenderList& renderList, Camera* pCamera)
{
  DrawRenderList<&Renderable::DrawShadowMap>(renderList, pCamera);
}

void SceneRenderer::DrawCubeShadowMap(const RenderList& renderList, Camera* pCamera)
{
  DrawRenderList<&Renderable::DrawCubeShadowMap>(renderList, pCamera);
}

void SceneRenderer::DrawCubeShadowMapArray(const RenderList& renderList, Camera* pCamera)
{
  DrawRenderList<&Renderable::DrawCubeShadowMapArray>(renderList, pCamera);
}

void SceneRenderer::DrawParabolicShadowMap(const RenderList& renderList, Camera* pCamera)
{
  DrawRenderList<&Renderable::DrawParabolicShadowMap>(renderList, pCamera);
}

void SceneRenderer::DrawASMLayerShadowMap(const RenderList& renderList, Camera* pCamera)
{
  DrawRenderList<&Renderable::DrawASMLayerShadowMap>(renderList, pCamera);
}

ID3D11Buffer* SceneRenderer::GetGlobalConstants(Camera* pCamera, DeviceContext11& dc)
{
  GlobalConst gc;
  gc.g_ViewProjection = pCamera->GetViewProjection();
  gc.g_ViewMat = pCamera->GetViewMatrix();
  gc.g_ProjMat = pCamera->GetProjection();
  gc.g_CameraUp = pCamera->GetUpVector();
  gc.g_CameraRight = pCamera->GetRightVector();
  gc.g_CameraPos = pCamera->GetPosition();
  return dc.GetConstantBuffers().Allocate(sizeof(gc), &gc, dc.DoNotFlushToDevice());
}

static __declspec(thread) struct TLSData
{
  DeviceContext11* pDC;
  unsigned ThreadID;
} g_TLS;

struct GlobalThreadData
{
  DeferredContext11* pDeferredContext;
};

static tbb::spin_mutex g_GlobalDataMutex;
static unsigned g_NActiveThreads = 0;
static std::vector<GlobalThreadData> g_GlobalThreadData;
static size_t g_WorkCounter[32];

class TBBObserver : public tbb::task_scheduler_observer
{
public:
  TBBObserver()
  {
    observe(true);
  }
  virtual void on_scheduler_entry(bool is_worker) override
  {
    _ASSERT(g_TLS.pDC==NULL);
    Log::Info("TBB attaches %s 0x%x\n", is_worker ? "worker" : "master", GetCurrentThreadId());
    GlobalThreadData gtd = { };
    if(is_worker)
    {
      gtd.pDeferredContext = new DeferredContext11();
      g_TLS.pDC = gtd.pDeferredContext;
    }
    else
    {
      g_TLS.pDC = &Platform::GetImmediateContext();
    }
    tbb::spin_mutex::scoped_lock lock(g_GlobalDataMutex);
    g_TLS.ThreadID = g_GlobalThreadData.size();
    g_GlobalThreadData.push_back(gtd);
    ++g_NActiveThreads;
  }
  virtual void on_scheduler_exit(bool is_worker) override
  {
    if(g_TLS.pDC==NULL)
      return;
    Log::Info("TBB detaches %s 0x%x\n", is_worker ? "worker" : "master", GetCurrentThreadId());
    g_TLS.pDC = NULL;
    tbb::spin_mutex::scoped_lock lock(g_GlobalDataMutex);
    unsigned threadID = g_TLS.ThreadID;
    delete g_GlobalThreadData[threadID].pDeferredContext;
    g_GlobalThreadData[threadID].pDeferredContext = NULL;
    --g_NActiveThreads;
  }
};

class TBBInit : public tbb::task_scheduler_init
{
public:
  TBBInit(int number_of_threads) : tbb::task_scheduler_init(number_of_threads) { }
};

void ParallelSceneRenderer::Init()
{
  m_TBBObserver = new TBBObserver();
  m_TBBInit = new TBBInit(4);
}

void ParallelSceneRenderer::Shutdown()
{
  delete m_TBBInit;
  while(g_NActiveThreads>0)
    Sleep(100);
  delete m_TBBObserver;
}

template<void (Renderable::*Draw)(DeviceContext11&)> void ParallelDrawRenderList(const SceneRenderer::RenderList& renderList, ID3D11Buffer* pBuffer)
{
  DeviceContext11& idc = Platform::GetImmediateContext();
  idc.PushRC();

  idc.VSSetConstantBuffer(GlobalConst::BANK, pBuffer);
  idc.GSSetConstantBuffer(GlobalConst::BANK, pBuffer);
  idc.PSSetConstantBuffer(GlobalConst::BANK, pBuffer);
  idc.HSSetConstantBuffer(GlobalConst::BANK, pBuffer);
  idc.DSSetConstantBuffer(GlobalConst::BANK, pBuffer);

  g_GlobalDataMutex.lock();
  for(auto it=g_GlobalThreadData.begin(); it!=g_GlobalThreadData.end(); ++it)
  {
    if(it->pDeferredContext!=NULL)
    {
      *static_cast<RenderContext11*>(it->pDeferredContext) = idc;
      it->pDeferredContext->VSSetConstantBuffer(GlobalConst::BANK, pBuffer);
      it->pDeferredContext->GSSetConstantBuffer(GlobalConst::BANK, pBuffer);
      it->pDeferredContext->PSSetConstantBuffer(GlobalConst::BANK, pBuffer);
      it->pDeferredContext->HSSetConstantBuffer(GlobalConst::BANK, pBuffer);
      it->pDeferredContext->DSSetConstantBuffer(GlobalConst::BANK, pBuffer);
    }
  }
  g_GlobalDataMutex.unlock();

  memset(g_WorkCounter, 0, sizeof(g_WorkCounter));
  tbb::parallel_for(tbb::blocked_range<size_t>(0, renderList.size()), [&](const tbb::blocked_range<size_t>& r)
  {
    DeviceContext11* pDC = g_TLS.pDC;
    for(size_t i=r.begin(); i!=r.end(); ++i)
      (renderList[i]->*Draw)(*pDC);
    unsigned threadID = g_TLS.ThreadID;
    g_WorkCounter[threadID % ARRAYSIZE(g_WorkCounter)] += r.size();
  });

  g_GlobalDataMutex.lock();
  for(auto it=g_GlobalThreadData.begin(); it!=g_GlobalThreadData.end(); ++it)
  {
    if(it->pDeferredContext!=NULL)
    {
      it->pDeferredContext->FinishCommandList();
      it->pDeferredContext->ExecuteCommandList();
    }
  }
  g_GlobalDataMutex.unlock();

  idc.GetConstantBuffers().Free(pBuffer);
  idc.PopRC();
}

void ParallelSceneRenderer::DrawPrePass(const RenderList& renderList, Camera* pCamera)
{
  _ASSERT(m_TBBInit!=NULL && m_TBBObserver!=NULL);
  ParallelDrawRenderList<&Renderable::DrawPrePass>(renderList, GetGlobalConstants(pCamera, Platform::GetImmediateContext()));
}

void ParallelSceneRenderer::DrawShadowMap(const RenderList& renderList, Camera* pCamera)
{
  _ASSERT(m_TBBInit!=NULL && m_TBBObserver!=NULL);
  ParallelDrawRenderList<&Renderable::DrawShadowMap>(renderList, GetGlobalConstants(pCamera, Platform::GetImmediateContext()));
}

void ParallelSceneRenderer::DrawCubeShadowMap(const RenderList& renderList, Camera* pCamera)
{
  _ASSERT(m_TBBInit!=NULL && m_TBBObserver!=NULL);
  ParallelDrawRenderList<&Renderable::DrawCubeShadowMap>(renderList, GetGlobalConstants(pCamera, Platform::GetImmediateContext()));
}

void ParallelSceneRenderer::DrawCubeShadowMapArray(const RenderList& renderList, Camera* pCamera)
{
  _ASSERT(m_TBBInit!=NULL && m_TBBObserver!=NULL);
  ParallelDrawRenderList<&Renderable::DrawCubeShadowMapArray>(renderList, GetGlobalConstants(pCamera, Platform::GetImmediateContext()));
}

void ParallelSceneRenderer::DrawParabolicShadowMap(const RenderList& renderList, Camera* pCamera)
{
  _ASSERT(m_TBBInit!=NULL && m_TBBObserver!=NULL);
  ParallelDrawRenderList<&Renderable::DrawParabolicShadowMap>(renderList, GetGlobalConstants(pCamera, Platform::GetImmediateContext()));
}

void ParallelSceneRenderer::DrawASMLayerShadowMap(const RenderList& renderList, Camera* pCamera)
{
  _ASSERT(m_TBBInit!=NULL && m_TBBObserver!=NULL);
  ParallelDrawRenderList<&Renderable::DrawASMLayerShadowMap>(renderList, GetGlobalConstants(pCamera, Platform::GetImmediateContext()));
}
