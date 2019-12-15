#include "PreCompile.h"
#include "Light.h"
#include "LightBuffer.h"
#include "Scene/Camera.h"
#include "Renderer/Renderer.h"
#include "Scene/Scene.h"
#include "Platform11/IABuffer11.h"
#include "ShaderCache/SimpleShader.h"
#include "PushBuffer/PreallocatedPushBuffer.h"
#include "../../Util/RadixSort.h"
#include "../../Util/DebugRenderer.h"

std::deque<ReflectiveCubeShadowMapPointLight::Task> ReflectiveCubeShadowMapPointLight::s_Tasks;

bool ReflectiveCubeShadowMapPointLightBatch::OnPlatformInit()
{
  HRESULT hr = m_PositionBuffer.Init(m_FaceSize, m_FaceSize, DXGI_FORMAT_R32G32B32A32_FLOAT);
  hr = SUCCEEDED(hr) ? m_FloodMap.Init(m_FaceSize, m_FaceSize, DXGI_FORMAT_R32_UINT) : hr;
  for(unsigned i=0; i<2; ++i)
  {
    hr = SUCCEEDED(hr) ? m_IDs[i].Init(m_FaceSize, m_FaceSize, DXGI_FORMAT_R32_TYPELESS) : hr;
    hr = SUCCEEDED(hr) ? m_IDs[i].AddRenderTargetView(DXGI_FORMAT_R32_FLOAT) : hr;
    hr = SUCCEEDED(hr) ? m_IDs[i].AddRenderTargetView(DXGI_FORMAT_R32_UINT) : hr;
    hr = SUCCEEDED(hr) ? m_IDs[i].AddShaderResourceView(DXGI_FORMAT_R32_FLOAT) : hr;
    hr = SUCCEEDED(hr) ? m_IDs[i].AddShaderResourceView(DXGI_FORMAT_R32_UINT) : hr;
  }
  for(unsigned i=0; i<6; ++i)
  {
    hr = SUCCEEDED(hr) ? m_AlbedoRT[i].Init(m_FaceSize, m_FaceSize, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) : hr;
    hr = SUCCEEDED(hr) ? m_GeomNormalRT[i].Init(m_FaceSize, m_FaceSize, DXGI_FORMAT_R8G8B8A8_UNORM) : hr;
    hr = SUCCEEDED(hr) ? m_DepthRT[i].Init(m_FaceSize, m_FaceSize, DXGI_FORMAT_R16_TYPELESS, 1, NULL, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL) : hr;
  }
  for(unsigned i=0; i<c_NReadbackTextures; ++i)
  {
    hr = SUCCEEDED(hr) ? m_IDReadback[i].Init(m_FaceSize, m_FaceSize, DXGI_FORMAT_R32_TYPELESS, 1, NULL, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ) : hr;
    hr = SUCCEEDED(hr) ? m_ColorReadback[i].Init(m_FaceSize, m_FaceSize, DXGI_FORMAT_R8G8B8A8_TYPELESS, 1, NULL, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ) : hr;
    hr = SUCCEEDED(hr) ? m_NormalReadback[i].Init(m_FaceSize, m_FaceSize, DXGI_FORMAT_R8G8B8A8_TYPELESS, 1, NULL, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ) : hr;
    hr = SUCCEEDED(hr) ? m_PositionReadback[i].Init(m_FaceSize, m_FaceSize, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, NULL, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ) : hr;
  }
  return SUCCEEDED(hr);
}

void ReflectiveCubeShadowMapPointLightBatch::OnPlatformShutdown()
{
  m_PositionBuffer.Clear();
  m_FloodMap.Clear();
  m_IDs[0].Clear();
  m_IDs[1].Clear();
  for(unsigned i=0; i<6; ++i)
  {
    m_AlbedoRT[i].Clear();
    m_GeomNormalRT[i].Clear();
    m_DepthRT[i].Clear();
  }
  for(unsigned i=0; i<c_NReadbackTextures; ++i)
  {
    m_IDReadback[i].Clear();
    m_ColorReadback[i].Clear();
    m_NormalReadback[i].Clear();
    m_PositionReadback[i].Clear();
  }
}

ReflectiveCubeShadowMapPointLight::ReflectiveCubeShadowMapPointLight(unsigned faceSize, SceneObject* pParent, SceneQTreeNode* pQTreeRoot, unsigned NumVLights, unsigned VLightSMapFaceSize) :
  CubeShadowMapPointLight(pParent, pQTreeRoot), m_Reflectivity(0.1f)
{
  ReflectiveCubeShadowMapPointLightBatch::Allocate(this, faceSize);
  s_Lights.insert(this);
  if(VLightSMapFaceSize>0)
  {
    for(unsigned i=0; i<NumVLights; ++i)
      m_SMapVLights.push_back(new HemisphericalCubeShadowMapPointLight(VLightSMapFaceSize, NULL, pQTreeRoot, true));
  }
  else
  {
    for(unsigned i=0; i<NumVLights; ++i)
      m_VLights.push_back(new HemisphericalPointLight(NULL, pQTreeRoot));
  }
  m_VLP.resize(NumVLights);
  m_NewVLP.resize(NumVLights);
  if(NumVLights>0)
    memset(&m_VLP[0], 0, sizeof(VirtualLightParam)*NumVLights);
}

ReflectiveCubeShadowMapPointLight::~ReflectiveCubeShadowMapPointLight()
{
  DeleteTasks();
  s_Lights.remove(this);
  std::for_each(m_SMapVLights.begin(), m_SMapVLights.end(), [] (HemisphericalCubeShadowMapPointLight* p) { delete p; });
  std::for_each(m_VLights.begin(), m_VLights.end(), [] (HemisphericalPointLight* p) { delete p; });
}

ReflectiveCubeShadowMapPointLightBatch::ReflectiveCubeShadowMapPointLightBatch(unsigned faceSize) : CubeShadowMapPointLightBatch(faceSize)
{
  Platform::Add(Platform::OnInitDelegate::from_method<ReflectiveCubeShadowMapPointLightBatch, &ReflectiveCubeShadowMapPointLightBatch::OnPlatformInit>(this), Platform::Object_Generic);
  Platform::Add(Platform::OnShutdownDelegate::from_method<ReflectiveCubeShadowMapPointLightBatch, &ReflectiveCubeShadowMapPointLightBatch::OnPlatformShutdown>(this), Platform::Object_Generic);
  s_Batches.insert(this);
}

ReflectiveCubeShadowMapPointLightBatch::~ReflectiveCubeShadowMapPointLightBatch()
{
  s_Batches.remove(this);
}

void ReflectiveCubeShadowMapPointLightBatch::Allocate(ReflectiveCubeShadowMapPointLight* pLight, unsigned faceSize)
{
  auto it = std::find_if(s_Batches.cbegin(), s_Batches.cend(), 
    [&] (ReflectiveCubeShadowMapPointLightBatch* b) -> bool { return b->m_FaceSize==faceSize && b->m_FreeRect.size()>0; });
  (it!=s_Batches.cend() ? *it : new ReflectiveCubeShadowMapPointLightBatch(faceSize))->Add(pLight);
}

class CubeMapVisibleObjects : public std::vector<SceneObject*>
{
public:
  void Find(Scene* pScene, Camera* pCamera, unsigned frameNumber)
  {
    LightBatch::ClearVisibilityInfo();
    m_TravelAgent.ProcessVisiblePrepare(pCamera->GetViewProjection(), pCamera->GetPosition(), frameNumber);
    m_TravelAgent.ProcessVisible<CubeMapVisibleObjects, &CubeMapVisibleObjects::TraversalCallback>(this, pScene->GetQTreeRoot());
  }

protected:
  SceneQTreeTraversal m_TravelAgent;

  finline void TraversalCallback(SceneObject* pObj)
  {
    push_back(pObj);
  }
};

void ReflectiveCubeShadowMapPointLightBatch::CopyFaceDepth(RenderTarget2D& depthRT, const Vec4i& rect, int faceIndex, DeviceContext11& dc)
{
  D3D11_VIEWPORT vp = { };
  vp.TopLeftX = float((faceIndex%3)*m_FaceSize + rect.x);
  vp.TopLeftY = float((faceIndex/3)*m_FaceSize + rect.y);
  vp.Width = vp.Height = float(m_FaceSize);
  vp.MaxDepth = 1.0f;
  dc.SetViewport(vp);

  static const D3D11_INPUT_ELEMENT_DESC c_InputDesc = {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0};
  static size_t s_DepthCopyShaderIndex = g_SimpleShaderCache.GetIndex(SimpleShaderDesc("_Shaders\\CopyDepth.shader", NULL, "_Shaders\\CopyDepth.shader", NULL, NULL, NULL, &c_InputDesc, 1));
  g_SimpleShaderCache.GetByIndex(s_DepthCopyShaderIndex).Bind();

  static const Vec4 c_ScreenQuadData[]=
  {
    Vec4(+1, -1, 1, 1),
    Vec4(-1, -1, 0, 1),
    Vec4(+1, +1, 1, 0),
    Vec4(-1, +1, 0, 0),
  };
  static StaticIABuffer<ARRAYSIZE(c_ScreenQuadData), sizeof(Vec4)> s_QuadVB(c_ScreenQuadData);
  dc.BindVertexBuffer(0, &s_QuadVB, 0);
  dc.SetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

  static size_t s_DepthStencilBlockIndex = Platform::GetDepthStencilCache().ConcurrentGetIndex(DepthStencilDesc11(true, D3D11_DEPTH_WRITE_MASK_ALL, D3D11_COMPARISON_ALWAYS));
  dc.SetDepthStencilState(&Platform::GetDepthStencilCache().ConcurrentGetByIndex(s_DepthStencilBlockIndex));

  dc.BindPS(0, &depthRT);
  dc.SetSamplerPS(0, &Platform::GetSamplerCache().GetByIndex(Platform::Sampler_Point_Clamp));

  dc.FlushToDevice()->Draw(4, 0);
}

void ReflectiveCubeShadowMapPointLightBatch::Update(SceneRenderer* pRenderer, Scene* pScene, ReflectiveCubeShadowMapPointLight* pLight, DeviceContext11& dc)
{
  _ASSERT(pLight->m_Batch==this);

  float lightRange = pLight->GetRange();
  Camera camera;
  camera.SetViewMatrix(Mat4x4::TranslationD3D(-pLight->GetPosition()));
  camera.SetProjection(Mat4x4::OrthoD3D(-lightRange, lightRange, -lightRange, lightRange, -lightRange, lightRange));

  static CubeMapVisibleObjects s_CubeMapVisible;
  s_CubeMapVisible.reserve(1024);
  s_CubeMapVisible.clear();
  s_CubeMapVisible.Find(pScene, &camera, pScene->GenerateTimeStamp());

  Mat4x4 projMat = GetProjectionMatrix(lightRange);
  pLight->m_SMapShaderData.x = projMat.e43;
  pLight->m_SMapShaderData.y = projMat.e33;
  pLight->m_SMapShaderData.z = (float)pLight->m_SMapRect.x/(float)m_DepthAtlas.GetDesc().Width;
  pLight->m_SMapShaderData.w = (float)pLight->m_SMapRect.y/(float)m_DepthAtlas.GetDesc().Height;

  camera.SetProjection(projMat);
  for(int i=0; i<6; ++i)
  {
    dc.PushRC();
    dc.BindRT(0, &m_AlbedoRT[i]);
    dc.BindRT(1, &m_GeomNormalRT[i]);
    dc.BindDepthStencil(&m_DepthRT[i]);
    m_DepthRT[i].SetViewport(dc);

    dc.ClearRenderTarget(0, Vec4::Zero());
    dc.ClearRenderTarget(1, Vec4::Zero());
    dc.ClearDepth(1.0f);

    camera.SetViewMatrix(pLight->GetViewMatrix(i));
    Frustum faceFrustum = Frustum::FromViewProjectionMatrixD3D(camera.GetViewProjection(), camera.GetViewProjectionInverse());
    SceneRenderer::RenderList s_RenderList;
    s_RenderList.reserve(1024);
    s_RenderList.clear();
    for(auto it=s_CubeMapVisible.begin(); it!=s_CubeMapVisible.end(); ++it)
    {
      SceneObject* pObj = *it;
      if(faceFrustum.IsIntersecting(pObj->GetOBB(), pObj->GetBSphereRadius()))
      {
        Renderable* pRenderable = pObj->PrepareToRender();
        if(pRenderable!=NULL)
          s_RenderList.push_back(pRenderable);
      }
    }
    pRenderer->DrawPrePass(s_RenderList, &camera);

    dc.RestoreRC();
    dc.UnbindRT(0);
    dc.BindDepthStencil(&m_DepthAtlas);

    CopyFaceDepth(m_DepthRT[i], pLight->m_SMapRect, i, dc);

    dc.PopRC();
  }

  pLight->DeleteTasks();
  ReflectiveCubeShadowMapPointLight::Task task;
  task.pLight = pLight;

  task.f = &ReflectiveCubeShadowMapPointLight::FloodFill;
  for(task.i=0; task.i<6; ++task.i)
    ReflectiveCubeShadowMapPointLight::s_Tasks.push_back(task);  

  task.f = &ReflectiveCubeShadowMapPointLight::ReadClusterData;
  for(task.i=0; task.i<6; ++task.i)
    ReflectiveCubeShadowMapPointLight::s_Tasks.push_back(task);

  task.f = &ReflectiveCubeShadowMapPointLight::UpdateShadowCastingVLight;
  for(task.i=0; task.i<pLight->m_SMapVLights.size(); ++task.i)
    ReflectiveCubeShadowMapPointLight::s_Tasks.push_back(task);

  task.f = &ReflectiveCubeShadowMapPointLight::UpdateVLights;
  ReflectiveCubeShadowMapPointLight::s_Tasks.push_back(task);

  if(pLight->m_NewVLP.size()>0)
    memset(&pLight->m_NewVLP[0], 0, sizeof(ReflectiveCubeShadowMapPointLight::VirtualLightParam)*pLight->m_NewVLP.size());
}

void ReflectiveCubeShadowMapPointLightBatch::FloodFill(int faceIndex, ReflectiveCubeShadowMapPointLight* pLight, DeviceContext11& dc)
{
  dc.PushRC();
  dc.BindRT(0, &m_FloodMap);
  dc.BindRT(1, &m_PositionBuffer);
  dc.UnbindDepthStencil();
  m_FloodMap.SetViewport(dc);

  static const Vec4 c_ScreenQuadData[]=
  {
    Vec4(+1, -1, 1, 1),
    Vec4(-1, -1, 0, 1),
    Vec4(+1, +1, 1, 0),
    Vec4(-1, +1, 0, 0),
  };
  static StaticIABuffer<ARRAYSIZE(c_ScreenQuadData), sizeof(Vec4)> s_QuadVB(c_ScreenQuadData);
  dc.BindVertexBuffer(0, &s_QuadVB, 0);
  dc.SetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

  dc.BindPS(0, &m_DepthRT[faceIndex]);
  dc.BindPS(1, &m_GeomNormalRT[faceIndex]);

  static const D3D11_INPUT_ELEMENT_DESC c_InputDesc = {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0};
  static size_t s_FloodMapShaderIndex = g_SimpleShaderCache.GetIndex(SimpleShaderDesc("_Shaders\\RSMFloodMap.shader", NULL, "_Shaders\\RSMFloodMap.shader", NULL, NULL, NULL, &c_InputDesc, 1));
  g_SimpleShaderCache.GetByIndex(s_FloodMapShaderIndex).Bind(dc);

  Mat4x4 invViewProj = Mat4x4::Inverse(pLight->GetViewMatrix(faceIndex)*GetProjectionMatrix(pLight->GetRange()));
  Mat4x4 invViewportTransform = Mat4x4::ScalingTranslationD3D(Vec3(2.0f/(float)m_FaceSize, -2.0f/(float)m_FaceSize, 1), Vec3(-1.0f, 1.0f, 0))*invViewProj;

  PreallocatedPushBuffer<> pb;
  pb.PushConstantPS(invViewportTransform, 0);
  Vec3 p00 = Vec3::Project(Vec3::Constant<-1,+1, 0>(), invViewProj);
  Vec3 p10 = Vec3::Project(Vec3::Constant<+1,+1, 0>(), invViewProj);
  Vec3 p01 = Vec3::Project(Vec3::Constant<-1,-1, 0>(), invViewProj);
  Vec3 dx = (p10 - p00)/(float)m_FaceSize;
  Vec3 dy = (p01 - p00)/(float)m_FaceSize;
  pb.PushConstantPS(p00 - pLight->GetPosition());
  pb.PushConstantPS(pLight->GetPosition());
  pb.PushConstantPS(dx);
  pb.PushConstantPS(dy);
  pb.PushConstantPS(pLight->m_ShaderData.Position);
  pb.Draw(4, 0);
  pb.Execute(dc);

  int DSTBuffer = 0;
  dc.RenderContext11::BindRT(0, m_IDs[DSTBuffer].GetRenderTargetView(0), NULL);
  dc.UnbindRT(1);
  dc.UnbindDepthStencil();

  static size_t s_FloodIDShaderIndex = g_SimpleShaderCache.GetIndex(SimpleShaderDesc("_Shaders\\RSMFloodID.shader", NULL, "_Shaders\\RSMFloodID.shader", NULL, NULL, NULL, &c_InputDesc, 1));
  g_SimpleShaderCache.GetByIndex(s_FloodIDShaderIndex).Bind(dc);

  pb.Clear();
  pb.PushConstantPS<unsigned>(m_FaceSize - 1, 0);
  pb.PushConstantPS<unsigned>(m_FaceSize - 1);
  pb.PushConstantPS(float(m_FaceSize));
  pb.Draw(4, 0);
  pb.Execute();

  const int N = 95;
  for(int i=0; i<=N; ++i)
  {
    DSTBuffer = 1 - DSTBuffer;
    dc.RenderContext11::BindRT(0, m_IDs[DSTBuffer].GetRenderTargetView(i==N ? 1 : 0), m_IDs[DSTBuffer].GetShaderResourceView(0));

    dc.BindPS(0, m_IDs[1 - DSTBuffer].GetShaderResourceView(0));
    dc.BindPS(1, &m_FloodMap);
    dc.SetSamplerPS(0, &Platform::GetSamplerCache().GetByIndex(Platform::Sampler_Point_Clamp));

    static size_t s_FloodFillShaderIndex = g_SimpleShaderCache.GetIndex(SimpleShaderDesc("_Shaders\\RSMFloodFill.shader", NULL, "_Shaders\\RSMFloodFill.shader", NULL, NULL, NULL, &c_InputDesc, 1));
    static size_t s_LastFloodFillShaderIndex = g_SimpleShaderCache.GetIndex(SimpleShaderDesc("_Shaders\\RSMFloodFill.shader", NULL, "_Shaders\\RSMFloodFill.shader?FINAL_PASS", NULL, NULL, NULL, &c_InputDesc, 1));
    g_SimpleShaderCache.GetByIndex(i==N ? s_LastFloodFillShaderIndex : s_FloodFillShaderIndex).Bind(dc);

    dc.FlushToDevice()->Draw(4, 0);
  }

  m_AlbedoRT[faceIndex].CopyTo(m_ColorReadback[faceIndex], dc);
  m_GeomNormalRT[faceIndex].CopyTo(m_NormalReadback[faceIndex], dc);
  m_IDs[DSTBuffer].CopyTo(m_IDReadback[faceIndex], dc);
  m_PositionBuffer.CopyTo(m_PositionReadback[faceIndex], dc);

  dc.PopRC();
}

void ReflectiveCubeShadowMapPointLightBatch::ReadClusterData(int faceIndex, ReflectiveCubeShadowMapPointLight* pLight, DeviceContext11& dc)
{
  _ASSERT(!(m_FaceSize & 3) && "must be multiple of four");

  const size_t NClusters = m_FaceSize*m_FaceSize;
  const size_t clusterBufferSize =sizeof(Vec4)*NClusters;
  Vec4* restrict pClusterColor = (Vec4*)Platform::AllocateScratchMemory(clusterBufferSize);
  Vec4* restrict pClusterNormal = (Vec4*)Platform::AllocateScratchMemory(clusterBufferSize);
  Vec4* restrict pClusterPosition = (Vec4*)Platform::AllocateScratchMemory(clusterBufferSize);
  memset(pClusterColor, 0, clusterBufferSize);
  memset(pClusterNormal, 0, clusterBufferSize);
  memset(pClusterPosition, 0, clusterBufferSize);

  D3D11_MAPPED_SUBRESOURCE msrColorReadback = { };
  D3D11_MAPPED_SUBRESOURCE msrNormalReadback = { };
  D3D11_MAPPED_SUBRESOURCE msrPositionReadback = { };
  D3D11_MAPPED_SUBRESOURCE msrIDReadback = { };
  const unsigned subResID = D3D11CalcSubresource(0, 0, 1);
  dc.DoNotFlushToDevice()->Map(m_ColorReadback[faceIndex].GetTexture2D(), subResID, D3D11_MAP_READ, 0, &msrColorReadback);
  dc.DoNotFlushToDevice()->Map(m_NormalReadback[faceIndex].GetTexture2D(), subResID, D3D11_MAP_READ, 0, &msrNormalReadback);
  dc.DoNotFlushToDevice()->Map(m_PositionReadback[faceIndex].GetTexture2D(), subResID, D3D11_MAP_READ, 0, &msrPositionReadback);
  dc.DoNotFlushToDevice()->Map(m_IDReadback[faceIndex].GetTexture2D(), subResID, D3D11_MAP_READ, 0, &msrIDReadback);

  const int* restrict pID = (int*)msrIDReadback.pData;
  const unsigned* restrict pColorSrc = (unsigned*)msrColorReadback.pData;
  const unsigned* restrict pNormalSrc = (unsigned*)msrNormalReadback.pData;
  const float* restrict pPositionSrc = (float*)msrPositionReadback.pData;
  for(unsigned i=0; i<m_FaceSize; ++i)
  {
    for(unsigned j=0; j<m_FaceSize; j+=4)
    {
      Vec4i id(&pID[j]);

      Mat4x4 pos(&pPositionSrc[j*4]);
      Vec4 w0 = Vec4::Swizzle<w,w,w,w>(pos.r[0]);
      Vec4 w1 = Vec4::Swizzle<w,w,w,w>(pos.r[1]);
      Vec4 w2 = Vec4::Swizzle<w,w,w,w>(pos.r[2]);
      Vec4 w3 = Vec4::Swizzle<w,w,w,w>(pos.r[3]);

      Mat4x4 color = RGBA8UN_to_RGBA32F_4T(&pColorSrc[j]);
      pClusterColor[id.x] += color.r[0]*w0;
      pClusterColor[id.y] += color.r[1]*w1;
      pClusterColor[id.z] += color.r[2]*w2;
      pClusterColor[id.w] += color.r[3]*w3;

      Mat4x4 normal = RGBA8SN_to_RGBA32F_4T(&pNormalSrc[j]);
      pClusterNormal[id.x] += normal.r[0]*w0;
      pClusterNormal[id.y] += normal.r[1]*w1;
      pClusterNormal[id.z] += normal.r[2]*w2;
      pClusterNormal[id.w] += normal.r[3]*w3;

      pClusterPosition[id.x] += Vec3::Point(pos.r[0])*w0;
      pClusterPosition[id.y] += Vec3::Point(pos.r[1])*w1;
      pClusterPosition[id.z] += Vec3::Point(pos.r[2])*w2;
      pClusterPosition[id.w] += Vec3::Point(pos.r[3])*w3;
    }
    pID = (int*)((char*)pID + msrIDReadback.RowPitch);
    pColorSrc = (unsigned*)((char*)pColorSrc + msrColorReadback.RowPitch);
    pNormalSrc = (unsigned*)((char*)pNormalSrc + msrNormalReadback.RowPitch);
    pPositionSrc = (float*)((char*)pPositionSrc + msrPositionReadback.RowPitch);
  }

  dc.DoNotFlushToDevice()->Unmap(m_ColorReadback[faceIndex].GetTexture2D(), subResID);
  dc.DoNotFlushToDevice()->Unmap(m_NormalReadback[faceIndex].GetTexture2D(), subResID);
  dc.DoNotFlushToDevice()->Unmap(m_PositionReadback[faceIndex].GetTexture2D(), subResID);
  dc.DoNotFlushToDevice()->Unmap(m_IDReadback[faceIndex].GetTexture2D(), subResID);

  const size_t NumVLights = pLight->m_VLP.size();
  struct SortStruct { float Key; unsigned ID; };
  SortStruct* pTmpSortBuf0 = (SortStruct*)Platform::AllocateScratchMemory(sizeof(SortStruct)*(NClusters + NumVLights));
  SortStruct* pTmpSortBuf1 = (SortStruct*)Platform::AllocateScratchMemory(sizeof(SortStruct)*(NClusters + NumVLights));
  static const unsigned c_WeightThreshold = FloatAsInt(0.0001f);
  SortStruct* restrict pSortBuf = pTmpSortBuf0;
  const unsigned* restrict pWeight = (unsigned*)&pClusterPosition[0].w;
  unsigned toSort = 0;
  for(size_t i=0; i<NClusters; ++i)
  {
    pSortBuf[toSort].ID = i;
    pSortBuf[toSort].Key = pClusterPosition[i].w;
    toSort += (*pWeight > c_WeightThreshold);
    pWeight = (unsigned*)((char*)pWeight + sizeof(Vec4));
  }
  for(unsigned i=0; i<NumVLights; ++i)
  {
    const ReflectiveCubeShadowMapPointLight::VirtualLightParam& vlp = pLight->m_NewVLP[i];
    pSortBuf[toSort + i].ID = NClusters + i;
    pSortBuf[toSort + i].Key = vlp.Falloff*vlp.Area;
  }
  pSortBuf = RadixSort_Descending<offsetof(SortStruct, Key), sizeof(pSortBuf[0].Key)>
    (toSort + NumVLights, pTmpSortBuf0, pTmpSortBuf1);

  static AlignedPODVector<ReflectiveCubeShadowMapPointLight::VirtualLightParam> s_VLP;
  s_VLP.resize(NumVLights);
  for(size_t i=0; i<NumVLights; ++i)
  {
    ReflectiveCubeShadowMapPointLight::VirtualLightParam& vlp = s_VLP[i];
    unsigned clusterID = pSortBuf[i].ID;
    if(clusterID<NClusters)
    {
      float formFactor = pClusterPosition[clusterID].w;
      float f = 1.0f/formFactor;
      vlp.Position = f*pClusterPosition[clusterID];
      vlp.Color = f*pClusterColor[clusterID];
      vlp.Normal = Vec3::Normalize(Vec3(pClusterNormal[clusterID]));

      Vec3 d = pLight->GetPosition() - vlp.Position;
      float distSq = Vec3::LengthSq(d);
      float attenuation = std::max(0.0f, 1.0f - distSq*pLight->m_ShaderData.Position.w);
      Vec3 L = d*Vec4::ApproxRsqrt(distSq);
      vlp.Falloff = attenuation*std::max(0.0f, Vec3::Dot(vlp.Normal, L));
      vlp.Area = formFactor/vlp.Falloff;
    }
    else
    {
      vlp = pLight->m_NewVLP[clusterID - NClusters];
    }
  }
  pLight->m_NewVLP.swap(s_VLP);
  Platform::FreeScratchMemory(pTmpSortBuf0);
  Platform::FreeScratchMemory(pTmpSortBuf1);
  Platform::FreeScratchMemory(pClusterColor);
  Platform::FreeScratchMemory(pClusterNormal);
  Platform::FreeScratchMemory(pClusterPosition);
}

void ReflectiveCubeShadowMapPointLight::Commit()
{
  __super::Commit();
  for(size_t i=0; i<m_VLights.size(); ++i)
    UpdateVirtualLightParam(m_VLights[i], m_VLP[i]);
  for(size_t i=0; i<m_SMapVLights.size(); ++i)
    UpdateVirtualLightParam(m_SMapVLights[i], m_VLP[i]);
}

void ReflectiveCubeShadowMapPointLight::DebugDrawVirtualLights(DebugRenderer& drm)
{
  drm.SetContourColor(Vec4(1.0f));
  for(auto it=m_VLP.begin(); it!=m_VLP.end(); ++it)
  {
    const VirtualLightParam& vlp = *it;
    drm.SetFillColor(Vec3::Point(vlp.Color));
    Vec3 t = GetArbitraryOrthogonalVector(vlp.Normal);
    Vec3 b = Vec3::Cross(t, vlp.Normal);
    Vec3 p = vlp.Position + 0.05f*vlp.Normal;
    float s = 0.1f*vlp.Falloff*vlp.Area;
    drm.PushQuad(Vec3(p - s*t - s*b), Vec3(p - s*t + s*b), Vec3(p + s*t + s*b), Vec3(p + s*t - s*b));
  }
}

void ReflectiveCubeShadowMapPointLight::UpdateShadowCastingVLight(unsigned lightIndex, SceneRenderer* pRenderer, Scene* pScene, DeviceContext11& dc)
{
  m_VLP[lightIndex] = m_NewVLP[lightIndex];
  UpdateVirtualLightParam(m_SMapVLights[lightIndex], m_VLP[lightIndex]);
  m_SMapVLights[lightIndex]->Update(pRenderer, pScene, dc);
}

void ReflectiveCubeShadowMapPointLight::UpdateVLights(unsigned, SceneRenderer*, Scene*, DeviceContext11&)
{
  for(size_t i=0; i<m_VLights.size(); ++i)
  {
    m_VLP[i] = m_NewVLP[i];
    UpdateVirtualLightParam(m_VLights[i], m_VLP[i]);
  }
}

void ReflectiveCubeShadowMapPointLight::FloodFill(unsigned faceIndex, SceneRenderer*, Scene*, DeviceContext11& dc)
{
  static_cast<ReflectiveCubeShadowMapPointLightBatch*>(m_Batch)->FloodFill(faceIndex, this, dc);
}

void ReflectiveCubeShadowMapPointLight::ReadClusterData(unsigned faceIndex, SceneRenderer*, Scene*, DeviceContext11& dc)
{
  static_cast<ReflectiveCubeShadowMapPointLightBatch*>(m_Batch)->ReadClusterData(faceIndex, this, dc);
}

void ReflectiveCubeShadowMapPointLight::DeleteTasks()
{
  for(auto it = s_Tasks.begin(); it!=s_Tasks.end();)
    it = (it->pLight==this) ? s_Tasks.erase(it) : (it + 1);
}

bool ReflectiveCubeShadowMapPointLight::Process(SceneRenderer* pRenderer, Scene* pScene, DeviceContext11& dc)
{
  if(!s_Tasks.empty())
  {
    const Task& task = s_Tasks.front();
    (task.pLight->*task.f)(task.i, pRenderer, pScene, dc);
    s_Tasks.pop_front();
  }
  return !s_Tasks.empty();
}
