#include "PreCompile.h"
#include "Light.h"
#include "LightBuffer.h"
#include "Shaders.h"
#include "Platform11/StructuredBuffer11.h"
#include "Scene/Camera.h"
#include "Renderer/Renderer.h"
#include "PushBuffer/PreallocatedPushBuffer.h"
#include "_Shaders/CubeMapGlobalConst.inc"
#include "../../Util/AlignedVector.h"
#include "../../Util/DebugRenderer.h"
#include "LargeConstantBuffer.h"

ptr_set<LightBatch, &LightBatch::GetListPos> LightBatch::s_Batches;
ptr_set<CubeShadowMapPointLightBatch, &CubeShadowMapPointLightBatch::GetListPos> CubeShadowMapPointLightBatch::s_Batches;
ptr_set<ReflectiveCubeShadowMapPointLightBatch, &ReflectiveCubeShadowMapPointLightBatch::GetListPos> ReflectiveCubeShadowMapPointLightBatch::s_Batches;
ptr_set<CubeShadowMapPointLight, &CubeShadowMapPointLight::GetListPos> CubeShadowMapPointLight::s_Lights;
ptr_set<ReflectiveCubeShadowMapPointLight, &ReflectiveCubeShadowMapPointLight::GetListPos> ReflectiveCubeShadowMapPointLight::s_Lights;

IMPLEMENT_MEMORY_POOL(PointLight, 1024);
IMPLEMENT_MEMORY_POOL(CubeShadowMapPointLight, 1024);
IMPLEMENT_MEMORY_POOL(HemisphericalPointLight, 1024);
IMPLEMENT_MEMORY_POOL(HemisphericalCubeShadowMapPointLight, 1024);
IMPLEMENT_MEMORY_POOL(ReflectiveCubeShadowMapPointLight, 1024);

PointLightBatch* g_PointLightBatch = new PointLightBatch();
HemisphericalPointLightBatch* g_HemisphericalPointLightBatch = new HemisphericalPointLightBatch();

PointLight::PointLight(SceneObject* pParent, SceneQTreeNode* pQTreeRoot) : 
  Light(pParent, pQTreeRoot, g_PointLightBatch)
{
  UpdateShaderData();
}

CubeShadowMapPointLight::CubeShadowMapPointLight(unsigned faceSize, SceneObject* pParent, SceneQTreeNode* pQTreeRoot, bool fastRender) :
  PointLight(pParent, pQTreeRoot), m_FastRender(fastRender), m_ArraySlice(-1)
{
  CubeShadowMapPointLightBatch::Allocate(this, faceSize);
  s_Lights.insert(this);
}

CubeShadowMapPointLight::CubeShadowMapPointLight(SceneObject* pParent, SceneQTreeNode* pQTreeRoot) :
  PointLight(pParent, pQTreeRoot), m_FastRender(false), m_ArraySlice(-1)
{
}

CubeShadowMapPointLight::~CubeShadowMapPointLight()
{
  _ASSERT(m_Batch!=g_PointLightBatch);
  static_cast<CubeShadowMapPointLightBatch*>(m_Batch)->Free(this);
  s_Lights.remove(this, true);
}

///////////////////////////////////////////////////////////////////////////////////////////

class StructuredBuffersPool : public BuffersPool<StructuredBuffersPool, StructuredBuffer, void*>
{
public:
  static const size_t c_MaxBufferSize = 65536;
  static const size_t c_ElementSize = sizeof(Vec4);
  bool Init() { return SUCCEEDED(__super::Init(NULL, c_MaxBufferSize, 8192, 8192)); }
  HRESULT InitBuffer(void*, StructuredBuffer& buf, size_t nElements) { return buf.Init(nElements, c_ElementSize, NULL, D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE); }
};

class LargeCBPool : public ConstantBuffersPool11
{
public:
  static const size_t c_Step = 4096;
  bool Init() { return SUCCEEDED(__super::Init(Platform::GetD3DDevice(), MAX_CB_SIZE, Platform::GetImmediateContext().GetConstantBuffers().GetMaxBufferSize() + c_Step, c_Step)); }
};

static class SharedResources
{
public:
  StructuredBuffersPool m_SBPool;
  LargeCBPool m_CBPool;

  ~SharedResources()
  {
    while(LightBatch::s_Batches.size()>0)
      delete LightBatch::s_Batches.front();
  }
  bool Init()
  {
    Platform::Add(Platform::OnShutdownDelegate::from_method<SharedResources, &SharedResources::OnPlatformShutdown>(this), Platform::Object_Generic);
    return Platform::Add(Platform::OnInitDelegate::from_method<SharedResources, &SharedResources::OnPlatformInit>(this), Platform::Object_Generic);
  }
  bool OnPlatformInit()
  {
    return m_SBPool.Init() && m_CBPool.Init();
  }
  void OnPlatformShutdown()
  {
    m_SBPool.Clear();
    m_CBPool.Clear();
  }
} g_SRes;

#define MIN_TILE_SIZE  64 // must be multiple of LIGHTING_QUAD_SIZE
#define TILE_SIZE_MASK ~(MIN_TILE_SIZE - 1)

HRESULT LightBuffer::Init(unsigned w, unsigned h)
{
  HRESULT hr = E_FAIL;
  if(g_SRes.Init())
  {
    m_Width = w;
    m_Height = h;
    m_QuadsWidth = (m_Width + LIGHTING_QUAD_SIZE - 1)/LIGHTING_QUAD_SIZE;
    m_QuadsHeight = (m_Height + LIGHTING_QUAD_SIZE - 1)/LIGHTING_QUAD_SIZE;
    m_VisibilityQuadsWidth = ((m_QuadsWidth + LIGHTING_QUAD_SIZE - 1)/LIGHTING_QUAD_SIZE)*LIGHTING_QUAD_SIZE;
    m_VisibilityQuadsHeight = ((m_QuadsHeight + LIGHTING_QUAD_SIZE - 1)/LIGHTING_QUAD_SIZE)*LIGHTING_QUAD_SIZE;
    hr = StructuredBuffer::Init(m_QuadsWidth*m_QuadsHeight*LIGHTING_GROUP_NTHREADS, sizeof(Vec4), NULL, D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE);
    hr = SUCCEEDED(hr) ? m_DepthBounds.Init(m_QuadsWidth, m_QuadsHeight, DXGI_FORMAT_R16G16_FLOAT) : hr;
    hr = SUCCEEDED(hr) ? m_VisibilityBuffer.Init(m_VisibilityQuadsWidth*m_VisibilityQuadsHeight*LIGHTING_GROUP_NTHREADS, sizeof(unsigned), NULL, D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE) : hr;
    hr = SUCCEEDED(hr) ? m_IndexTexture.Init((m_Width + MIN_TILE_SIZE - 1)/MIN_TILE_SIZE, (m_Height + MIN_TILE_SIZE - 1)/MIN_TILE_SIZE, DXGI_FORMAT_R32G32_UINT, 1, NULL, D3D11_USAGE_DYNAMIC, D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_WRITE) : hr;
  }
  return hr;
}

void LightBuffer::Clear()
{
  StructuredBuffer::Clear();
  m_DepthBounds.Clear();
  m_VisibilityBuffer.Clear();
  m_IndexTexture.Clear();
}

void LightBuffer::DepthReduction(RenderTarget2D* pDepthRT, RenderTarget2D* pGeomNormalBuffer, Camera* pCamera, DeviceContext11& dc)
{
  dc.PushRC();

  dc.BindRT(0, &m_DepthBounds);
  dc.UnbindDepthStencil();
  m_DepthBounds.SetViewport(dc);

  dc.BindPS(0, pDepthRT);
  dc.BindPS(1, pGeomNormalBuffer);

  static const D3D11_INPUT_ELEMENT_DESC c_InputDesc = {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0};
  static size_t s_ShaderIndex = g_SimpleShaderCache.GetIndex(SimpleShaderDesc("_Shaders\\LightingDepthReduction.shader", NULL, "_Shaders\\LightingDepthReduction.shader", NULL, NULL, NULL, &c_InputDesc, 1));
  g_SimpleShaderCache.GetByIndex(s_ShaderIndex).Bind(dc);

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

  PreallocatedPushBuffer<> pb;
  pb.PushConstantPS(Mat4x4::Transpose(pCamera->GetProjection()).r[2], 0);
  pb.Draw(4, 0);
  pb.Execute(dc);

  dc.PopRC();
}

static const Vec2 c_Split_C0(0.5f, 0);
static const Vec2 c_Split_C1(0, 0.5f);

void LightBuffer::SplitTile(Tile& tileToSplit, AlignedPODVector<Tile>& tiles, std::vector<unsigned>& IB, const AABB2D* pProjBBox)
{
  Vec4 bboxMin = tileToSplit.BBox.GetMin();
  Vec4 bboxMax = tileToSplit.BBox.GetMax();
  Vec4 bboxSize = bboxMax - bboxMin;
  Vec4 mask = Vec4::CmpGreater(Vec4::Swizzle<x,x,x,x>(bboxSize), Vec4::Swizzle<y,y,y,y>(bboxSize));
  Vec4 newBBoxMin = bboxMin + bboxSize*Vec4::Select(mask, c_Split_C0, c_Split_C1);
  newBBoxMin = Vec4i::Convert(Vec4i::Round(newBBoxMin) & Vec4i(TILE_SIZE_MASK));

  Tile newTiles[2];
  newTiles[0].BBox = AABB2D(bboxMin, bboxMin + bboxMax - newBBoxMin);
  newTiles[0].FirstLightIndex = tileToSplit.FirstLightIndex;
  newTiles[0].NLights = 0;
  newTiles[0].MergeBufferOffset = 0;

  newTiles[1].BBox = AABB2D(newBBoxMin, bboxMax);
  newTiles[1].FirstLightIndex = IB.size();
  newTiles[1].NLights = 0;
  newTiles[1].MergeBufferOffset = 0;

  IB.resize(newTiles[1].FirstLightIndex + tileToSplit.NLights);
  const unsigned* pSrc = &IB[tileToSplit.FirstLightIndex];
  unsigned* pDst[2] = { &IB[newTiles[0].FirstLightIndex], &IB[newTiles[1].FirstLightIndex] };
  for(unsigned i=0; i<tileToSplit.NLights; ++i, ++pSrc)
  {
    unsigned lightIndex = *pSrc;
    *pDst[0] = lightIndex;
    *pDst[1] = lightIndex;
    bool bInsersecting[2];
    bInsersecting[0] = AABB2D::IsIntersecting(pProjBBox[lightIndex], newTiles[0].BBox);
    bInsersecting[1] = AABB2D::IsIntersecting(pProjBBox[lightIndex], newTiles[1].BBox);
    pDst[0] += bInsersecting[0]; newTiles[0].NLights += bInsersecting[0];
    pDst[1] += bInsersecting[1]; newTiles[1].NLights += bInsersecting[1];
  }
  IB.resize(newTiles[1].FirstLightIndex + newTiles[1].NLights);

  tileToSplit = newTiles[0];
  tiles.push_back(newTiles[1]);
}

void LightBuffer::BuildTiles(unsigned NLights, const AABB2D* pProjBBox,
                             AlignedPODVector<Tile>& tiles, std::vector<unsigned>& IB,
                             DebugRenderer* pDRM)
{
  Tile rootTile;
  rootTile.BBox = AABB2D(Vec2::Zero(), Vec2(float((m_Width + MIN_TILE_SIZE - 1 ) & TILE_SIZE_MASK), float((m_Height + MIN_TILE_SIZE - 1 ) & TILE_SIZE_MASK)));
  rootTile.FirstLightIndex = 0;
  rootTile.NLights = NLights;
  rootTile.MergeBufferOffset = 0;
  tiles.resize(1);
  tiles[0] = rootTile;

  IB.resize(NLights);
  unsigned* pDst = &IB[0];
  for(unsigned i=0; i<NLights; ++i)
    *pDst++ = i;

  const int c_MaxSplits = 64;
  for(int i=0; i<c_MaxSplits; ++i)
  {
    unsigned lightsToSplit = 0;
    Tile* pTileToSplit = NULL;
    for(auto it=tiles.cbegin(); it!=tiles.cend(); ++it)
    {
      Tile& testTile = *it;
      if(!(testTile.BBox.Size()<=Vec2(MIN_TILE_SIZE)) && testTile.NLights>lightsToSplit)
      {
        pTileToSplit = &testTile;
        lightsToSplit = testTile.NLights;
      }
    }

    if(pTileToSplit==NULL)
      break;

    SplitTile(*pTileToSplit, tiles, IB, pProjBBox);
  }

  if(pDRM!=NULL)
  {
    static const Vec4 s_Colors[] =
    {
      Vec4(1.0f, 0.0f, 0.0f, 0.2f),
      Vec4(0.0f, 1.0f, 0.0f, 0.2f),
      Vec4(0.0f, 0.0f, 1.0f, 0.2f),
      Vec4(1.0f, 1.0f, 0.0f, 0.1f),
      Vec4(1.0f, 0.0f, 1.0f, 0.1f),
      Vec4(0.0f, 1.0f, 1.0f, 0.1f),
      Vec4(1.0f, 1.0f, 1.0f, 0.05f),
    };
    pDRM->SetContourColor(Vec4::Zero());
    for(size_t i=0; i<tiles.size(); ++i)
    {
      Tile& tile = tiles[i];
      Vec4 color = s_Colors[i % ARRAYSIZE(s_Colors)];
      if(tile.NLights>MAX_VISIBLE_LIGHTS_PER_QUAD) color.w = 1.0f;
      pDRM->SetFillColor(color);
      const AABB2D& aabb = tile.BBox;
      pDRM->PushQuad(Vec2(aabb.x, aabb.w), Vec2(aabb.x, aabb.y), Vec2(aabb.z, aabb.y), Vec2(aabb.z, aabb.w));
    }
  }
}

bool LightBuffer::BuildIndex(unsigned iPass, const Vec4* pCullData,
                             AlignedPODVector<Tile>& tiles, std::vector<unsigned>& IB,
                             const StructuredBuffer** ppBuffer, DeviceContext11& dc)
{
  bool needAnotherPass = false;
  Vec4* restrict pTemp = (Vec4*)Platform::AllocateScratchMemory(sizeof(Vec4)*StructuredBuffersPool::c_MaxBufferSize);
  size_t readPos=0, toWrite=0;
  for(auto it=tiles.cbegin(); it!=tiles.cend(); ++it)
  {
    Tile& tile = *it;
    int currentPassNLights = tile.NLights - iPass*MAX_VISIBLE_LIGHTS_PER_QUAD;
    if(currentPassNLights>MAX_VISIBLE_LIGHTS_PER_QUAD)
    {
      currentPassNLights = MAX_VISIBLE_LIGHTS_PER_QUAD;
      needAnotherPass = true;
    }
    tile.CurrentPassNLights = 0;
    if(currentPassNLights>0)
    {
      tile.CurrentPassNLights = currentPassNLights;
      tile.MergeBufferOffset = readPos;
      readPos += tile.CurrentPassNLights;

      const unsigned* pIB = &IB[tile.FirstLightIndex + iPass*MAX_VISIBLE_LIGHTS_PER_QUAD];
      for(int i=0; i<currentPassNLights; ++i)
        pTemp[toWrite++] = pCullData[*pIB++];
    }
  }
  const StructuredBuffer* pBuffer = g_SRes.m_SBPool.Allocate(toWrite);
  void* pMem = pBuffer->Map<void>(D3D11_MAP_WRITE_DISCARD, 0, dc);
  memcpy(pMem, pTemp, toWrite*sizeof(Vec4));
  pBuffer->Unmap(dc);
  Platform::FreeScratchMemory(pTemp);

  const unsigned lineWidth = m_IndexTexture.GetDesc().Width*2;
  const size_t lineSize = lineWidth*sizeof(unsigned);
  const size_t indexTextureBufferSize = lineSize*m_IndexTexture.GetDesc().Height;
  unsigned* pIndexTextureData = (unsigned*)alloca(indexTextureBufferSize);
  memset(pIndexTextureData, 0, indexTextureBufferSize);
  for(auto it=tiles.cbegin(); it!=tiles.cend(); ++it)
  {
    Tile& tile = *it;
    if(tile.CurrentPassNLights>0)
    {
      Vec4i intAABB = Vec4i::Round(tile.BBox/MIN_TILE_SIZE);
      _ASSERT(Vec4i::Convert(intAABB)*MIN_TILE_SIZE==tile.BBox);
      unsigned* pLine = &pIndexTextureData[intAABB.y*lineWidth + intAABB.x*2];
      for(int y=intAABB.y; y<intAABB.w; ++y)
      {
        unsigned* pDst = pLine;
        for(int x=intAABB.x; x<intAABB.z; ++x)
        {
          pDst[0] = tile.CurrentPassNLights;
          pDst[1] = tile.MergeBufferOffset;
          pDst+=2;
        }
        pLine += lineWidth;
      }
    }
  }

  D3D11_MAPPED_SUBRESOURCE msr;
  if(SUCCEEDED(dc.DoNotFlushToDevice()->Map(m_IndexTexture.GetTexture2D(), 0, D3D11_MAP_WRITE_DISCARD, 0, &msr)))
  {
    char* pDst = (char*)msr.pData;
    char* pSrc = (char*)pIndexTextureData;
    for(unsigned y=0; y<m_IndexTexture.GetDesc().Height; ++y)
    {
      memcpy(pDst, pSrc, lineSize);
      pSrc += lineSize;
      pDst += msr.RowPitch;
    }
    dc.DoNotFlushToDevice()->Unmap(m_IndexTexture.GetTexture2D(), 0);
  }

  *ppBuffer = pBuffer;
  return needAnotherPass;
}

static const IntegerMask c_BBox_C0 = {0x80000000, 0, 0, 0x3f800000};
static const IntegerMask c_BBox_C3 = {0, 0, 0x80000000, 0x80000000};
static const IntegerMask c_BBox_C5 = {0, 0x80000000, 0x80000000, 0};
static const Vec4 c_BBox_C1(0.5f, -0.5f, 0.5f, -0.5f);
static const Vec4 c_BBox_C2(1.0f, 0, 0, 1.0f);
static const Vec4 c_BBox_C4(0.5f, 0.5f, 0, 0);

bool LightBuffer::Render(RenderTarget2D* pNormalBuffer, RenderTarget2D* pDepthRT, RenderTarget2D* pGeomNormalBuffer, Camera* pCamera, DebugRenderer* pDRM, DeviceContext11& dc)
{
  DepthReduction(pDepthRT, pGeomNormalBuffer, pCamera, dc);

  Vec4 bufferSize = Vec4i::Convert(Vec4i(m_Width, m_Height, m_Width, m_Height));
  const Mat4x4& projMat = pCamera->GetProjection();
  Vec4 c0 = Vec4(projMat.e11, projMat.e22, projMat.e11, projMat.e22)*c_BBox_C1*bufferSize;
  Vec4 c1 = Vec4(0.5f)*bufferSize;
  Vec4 c2 = c_BBox_C4*bufferSize;
  Vec4 fullScreenBox = Vec4::Shuffle<x,y,z,w>(Vec4::Zero(), bufferSize);
  Vec4 fullScreenBCircle = 0.5f*Vec4::Shuffle<x,y,z,w>(bufferSize, Vec2::Length(Vec2(bufferSize)));

  static AlignedPODVector<Vec4> s_CullData;
  static AlignedPODVector<AABB2D> s_ProjBBox;
  size_t NLightsRendered = 0;
  for(auto it=LightBatch::s_Batches.begin(); it!=LightBatch::s_Batches.end(); ++it)
  {
    const std::vector<Light*>& toRender = (*it)->m_ToRender;
    if(toRender.size()>0)
    {
      s_CullData.resize(toRender.size());
      s_ProjBBox.resize(toRender.size());
      Vec4* pCullData = &s_CullData.front().as_type();
      Vec4* pProjBBox = &s_ProjBBox.front().as_type();
      int lightIndex = 0;
      for(auto iit=toRender.begin(); iit!=toRender.end(); ++iit, ++pCullData, ++pProjBBox, ++lightIndex)
      {
        Light* pLight = *iit;

        // This computes bounding circle of bounding sphere's projection onto the screen. 
        // As sphere's projection is an ellipse, it computes two points on major axis,
        // and then takes their average as circle's center, and half-distance as circle's
        // radius. The result, in viewport coordinates, is in BCircle. Circle's bounding box,
        // also in viewport coordinates, is in *pProjBBox.
        Vec3 viewPos = pLight->GetPosition()*pCamera->GetViewMatrix();
        float R = pLight->GetBSphereRadius();
        Vec4 r0 = viewPos*viewPos;
        Vec4 r1 = Vec4::Abs(r0 + Vec4::Swizzle<y,x,w,w>(r0) + Vec4::Swizzle<z,w,w,w>(r0));
        Vec4 r2 = Vec4::ApproxRsqrt(r1);
        Vec4 r3 = R*Vec4::Swizzle<x,x,x,x>(r2);
        Vec4 sincos = Vec4::ApproxSqrt(Vec4::Abs(c_BBox_C1*r3 + Vec4(0.5f)));
        Vec4 r4 = (Vec4::Swizzle<y,x,w,w>(viewPos)*Vec4::Swizzle<y,y,y,y>(r2)) ^ c_BBox_C0.f;
        Vec4 axis = Vec4::Select(Vec4::CmpGreater(Vec4::Swizzle<y,y,y,y>(r1), Vec4(1e-7f)), r4, c_BBox_C2);
        Quat rm = axis*Vec4::Swizzle<y,y,x,x>(sincos);
        Vec3 dir = -r3*viewPos;
        Vec4 p0 = viewPos + Quat::Transform(dir, rm);
        Vec4 p1 = viewPos + Quat::Transform(dir, Quat::Conjugate(rm));
        Vec4 pz = Vec4::Shuffle<z,z,z,z>(p0, p1);
        Vec4 pp = Vec4::Shuffle<x,y,x,y>(p0, p1)*Vec4::Rcp(pz);
        Vec4 r5 = 0.5f*(pp + (Vec4::Swizzle<z,w,x,y>(pp) ^ c_BBox_C3.f));
        Vec4 r6 = r5*r5;
        Vec4 r7 = Vec4::ApproxSqrt(Vec4::Swizzle<z,z,z,z>(r6) + Vec4::Swizzle<w,w,w,w>(r6));

        Vec4 isInside = Vec4::CmpLess(Vec4::Swizzle<z,z,z,z>(Vec4::Min(p0, p1)), pCamera->GetNear()) | Vec4::CmpGreater(r3, Vec4(1.0f));
        *pProjBBox = Vec4::Select(isInside, fullScreenBox, AABB2D(c0*(Vec4::Swizzle<x,y,x,y>(r5) - (r7 ^ c_BBox_C5.f)) + c1));
        Vec4i BCircle = Vec4i::Round(Vec4::Select(isInside, fullScreenBCircle, c0*Vec4::Shuffle<x,y,x,x>(r5, r7) + c2));

        Vec4i shaderData = BCircle << 16;
        shaderData.x |= BCircle.z; // pack bcircle radius
        shaderData.y |= lightIndex; // ... and index
        *pCullData = Vec4i::Cast(shaderData);
        pCullData->z = viewPos.z - R; // light's depth bounds
        pCullData->w = viewPos.z + R;
      }

      static AlignedPODVector<Tile> s_Tiles;
      static std::vector<unsigned> s_IB;
      BuildTiles(lightIndex, &s_ProjBBox.front().as_type(), s_Tiles, s_IB, pDRM);
      bool needAnotherPass = true;
      for(unsigned iPass=0; needAnotherPass; ++iPass)
      {
        const StructuredBuffer* pCullDataBuffer;
        needAnotherPass = BuildIndex(iPass, &s_CullData.front().as_type(), s_Tiles, s_IB, &pCullDataBuffer, dc);
        (*it)->Draw(this, pCullDataBuffer, pNormalBuffer, pDepthRT, pGeomNormalBuffer, pCamera, (NLightsRendered>0) | (iPass>0), dc);
        g_SRes.m_SBPool.Free(pCullDataBuffer);
      }
      NLightsRendered += toRender.size();

      if(pDRM!=NULL)
      {
        static const Vec4 c_Alpha(0, 0, 0, 0.3f);
        pDRM->SetFillColor(Vec4::Zero());
        Vec4* pProjBBox = &s_ProjBBox.front().as_type();
        for(auto it=toRender.begin(); it!=toRender.end(); ++it, ++pProjBBox)
        {
          pDRM->SetContourColor(c_Alpha | Vec3((*it)->GetColor()));
          pDRM->PushQuad(Vec2(pProjBBox->x, pProjBBox->w), Vec2(pProjBBox->x, pProjBBox->y), Vec2(pProjBBox->z, pProjBBox->y), Vec2(pProjBBox->z, pProjBBox->w));
        }
      }
    }
  }
  return NLightsRendered>0;
}

///////////////////////////////////////////////////////////////////////////////////////////

LargeConstantBuffer::LargeConstantBuffer(size_t dataSize, const void* pData, DeviceContext11& dc)
{
  m_Pool = dataSize<=dc.GetConstantBuffers().GetMaxBufferSize() ? &dc.GetConstantBuffers() : &g_SRes.m_CBPool;
  m_Buffer = m_Pool->Allocate(dataSize, pData, dc.DoNotFlushToDevice());
}

LargeConstantBuffer::~LargeConstantBuffer()
{
  m_Pool->Free(m_Buffer);
}

void LightBuffer::RunCulling(PushBuffer& pb, const StructuredBuffer* pCullDataBuffer, DeviceContext11& dc)
{
  dc.BindUA(0, &m_VisibilityBuffer);

  dc.BindCS(0, &m_DepthBounds);
  dc.BindCS(1, &m_IndexTexture);
  dc.BindCS(2, pCullDataBuffer);

  pb.PushConstantCS(m_VisibilityQuadsWidth);

  pb.Dispatch(m_VisibilityQuadsWidth/LIGHTING_QUAD_SIZE, m_VisibilityQuadsHeight/LIGHTING_QUAD_SIZE, 1);
  pb.Execute(dc);
}

inline Mat4x4 GetInvViewportTransform(Camera* pCamera, int viewportWidth, int viewportHeight)
{
  Mat4x4 invViewportTransform = Mat4x4::ScalingTranslationD3D(Vec3(2.0f/(float)viewportWidth, -2.0f/(float)viewportHeight, 1), Vec3(-1.0f, 1.0f, 0));
  return invViewportTransform*pCamera->GetViewProjectionInverse();
}

void LightBuffer::RunLighting(PushBuffer& pb, RenderTarget2D* pNormalBuffer, RenderTarget2D* pDepthRT,
                              RenderTarget2D* pGeomNormalBuffer, Camera* pCamera, DeviceContext11& dc)
{
  dc.BindUA(0, this);

  dc.BindCS(0, &m_VisibilityBuffer);
  dc.BindCS(1, pNormalBuffer);
  dc.BindCS(2, pDepthRT);
  dc.BindCS(3, pGeomNormalBuffer);

  pb.PushConstantCS(GetInvViewportTransform(pCamera, m_Width, m_Height));
  pb.PushConstantCS(m_QuadsWidth);
  pb.PushConstantCS(m_VisibilityQuadsWidth);

  pb.Dispatch(m_QuadsWidth, m_QuadsHeight, 1);
  pb.Execute(dc);
}

const ShaderObject& LightBatch::GetShader(const LightingShaderFlags& f)
{
  static LightingShaderCache s_Cache(64);
  return s_Cache.Get(f);
}

const ShaderObject& LightBatch::GetShader(const LightingQuadShaderFlags& f)
{
  static LightingQuadShaderCache s_Cache(64);
  return s_Cache.Get(f);
}

void PointLightBatch::Draw(LightBuffer* pLightBuffer,
                           const StructuredBuffer* pCullDataBuffer,
                           RenderTarget2D* pNormalBuffer, RenderTarget2D* pDepthRT,
                           RenderTarget2D* pGeomNormalBuffer, Camera* pCamera, 
                           bool bBlending, DeviceContext11& dc)
{
  LightingShaderFlags lsf;
  lsf.BLENDING = bBlending;
  PointLightBatch::Draw(lsf, pLightBuffer, pCullDataBuffer, pNormalBuffer, pDepthRT, pGeomNormalBuffer, pCamera, dc);
}

void PointLightBatch::Draw(const LightingShaderFlags& lsf, LightBuffer* pLightBuffer,
                           const StructuredBuffer* pCullDataBuffer,
                           RenderTarget2D* pNormalBuffer, RenderTarget2D* pDepthRT,
                           RenderTarget2D* pGeomNormalBuffer, Camera* pCamera,
                           DeviceContext11& dc)
{
  dc.PushRC();
  dc.UnbindRT(0);
  dc.UnbindDepthStencil();

  LightingQuadShaderFlags lqf;
  lqf.TGSM_WORKAROUND = Platform::GetFeatureLevel()<D3D_FEATURE_LEVEL_11_0; // temp fix for NVIDIA driver problem
  GetShader(lqf).Bind(dc);

  PreallocatedPushBuffer<> pb;
  pb.PushConstantCS(NULL, 0, 0);
  pLightBuffer->RunCulling(pb, pCullDataBuffer, dc);

  GetShader(lsf).Bind(dc);

  size_t NLights = std::min(m_ToRender.size(), size_t(MAX_LIGHTS_PER_PASS));
  if(m_ToRender.size()!=NLights)
    m_ToRender.resize(NLights);

  PointLightShaderData* pShaderData = (PointLightShaderData*)Platform::AllocateScratchMemory(sizeof(PointLightShaderData)*NLights);
  Light** ppLight = &m_ToRender.front();
  for(size_t i=0; i<NLights; ++i, ++ppLight)
    pShaderData[i] = static_cast<PointLight*>(*ppLight)->m_ShaderData;

  LargeConstantBuffer shaderDataCB(sizeof(*pShaderData)*NLights, pShaderData, dc);
  dc.CSSetConstantBuffer(PointLightsShaderData::BANK, shaderDataCB.GetBuffer());
  Platform::FreeScratchMemory(pShaderData);

  pb.Clear();
  pb.PushConstantCS(NULL, 0, 0);
  pLightBuffer->RunLighting(pb, pNormalBuffer, pDepthRT, pGeomNormalBuffer, pCamera, dc);

  dc.PopRC();
}

///////////////////////////////////////////////////////////////////////////////////////////

CubeShadowMapPointLightBatch::CubeShadowMapPointLightBatch(unsigned faceSize) : 
  m_NAllocated(0), m_FaceSize(faceSize), m_RectWidth(faceSize*3), m_RectHeight(faceSize*2), m_PCFKernelSize(faceSize>=256 ? 3 : 1)
{
  Platform::Add(Platform::OnInitDelegate::from_method<CubeShadowMapPointLightBatch, &CubeShadowMapPointLightBatch::OnPlatformInit>(this), Platform::Object_Generic);
  Platform::Add(Platform::OnShutdownDelegate::from_method<CubeShadowMapPointLightBatch, &CubeShadowMapPointLightBatch::OnPlatformShutdown>(this), Platform::Object_Generic);

  Vec4i d;
  d.z = m_RectWidth;
  d.w = m_RectHeight;
  for(d.y=0; d.y<(int)m_DepthAtlas.GetDesc().Height; d.y+=m_RectHeight)
    for(d.x=0; d.x<(int)m_DepthAtlas.GetDesc().Width; d.x+=m_RectWidth)
      m_FreeRect.push_back(d);

  for(unsigned i=0; i<m_CBMArray.GetDesc().ArraySize; i+=6)
    m_FreeSlice.push_back(i);

  s_Batches.insert(this);
}

CubeShadowMapPointLightBatch::~CubeShadowMapPointLightBatch()
{
  _ASSERT(!m_NAllocated && "some lights are still assigned to this batch");
  s_Batches.remove(this);
}

bool CubeShadowMapPointLightBatch::OnPlatformInit()
{
  const unsigned c_MaxPageSize = 4096;
  unsigned gridW = c_MaxPageSize/m_RectWidth;
  unsigned gridH = c_MaxPageSize/m_RectHeight;
  HRESULT hr = S_OK;
  if(m_PCFKernelSize==1 && Platform::GetFeatureLevel()>=D3D_FEATURE_LEVEL_11_0)
  {
    const unsigned c_CubemapsMax = D3D11_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION/6;
    unsigned m_NumCBMs = std::min(c_CubemapsMax, gridW*gridH);
    hr = m_CBMArray.Init(m_FaceSize, m_FaceSize, DXGI_FORMAT_R16_TYPELESS, 1, NULL, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL, 0, D3D11_RESOURCE_MISC_TEXTURECUBE, m_NumCBMs*6);
  }
  else
  {
    hr = m_DepthAtlas.Init(gridW*m_RectWidth, gridH*m_RectHeight, DXGI_FORMAT_R16_TYPELESS, 1, NULL, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL);
  }
  return SUCCEEDED(hr);
}

void CubeShadowMapPointLightBatch::OnPlatformShutdown()
{
  m_DepthAtlas.Clear();
  m_CBMArray.Clear();
}

void CubeShadowMapPointLightBatch::Allocate(CubeShadowMapPointLight* pLight, unsigned faceSize)
{
  auto it = std::find_if(s_Batches.cbegin(), s_Batches.cend(),
    [&] (CubeShadowMapPointLightBatch* b) -> bool { return b->m_FaceSize==faceSize && std::max(b->m_FreeRect.size(), b->m_FreeSlice.size())>0; });
  (it!=s_Batches.cend() ? *it : new CubeShadowMapPointLightBatch(faceSize))->Add(pLight);
}

void CubeShadowMapPointLightBatch::Add(CubeShadowMapPointLight* pLight)
{
  if(m_CBMArray.GetDesc().ArraySize>0)
  {
    pLight->m_ArraySlice = m_FreeSlice.back();
    m_FreeSlice.pop_back();
  }
  else
  {
    pLight->m_SMapRect = m_FreeRect.back();
    m_FreeRect.pop_back();
  }
  pLight->m_Batch = this;
  ++m_NAllocated;
}

void CubeShadowMapPointLightBatch::Free(CubeShadowMapPointLight* pLight)
{
  if(pLight->m_ArraySlice<0)
    m_FreeRect.push_back(pLight->m_SMapRect);
  else
    m_FreeSlice.push_back(pLight->m_ArraySlice);
  pLight->m_ArraySlice = -1;
  pLight->m_Batch = g_PointLightBatch;
  --m_NAllocated;
}

static const D3DXFLOAT16 c_CubeIndexData[] =
{
  D3DXFLOAT16(-3), D3DXFLOAT16(-1), D3DXFLOAT16(+1), D3DXFLOAT16(0),
  D3DXFLOAT16(+3), D3DXFLOAT16(-3), D3DXFLOAT16(-1), D3DXFLOAT16(0),
  D3DXFLOAT16(+1), D3DXFLOAT16(+5), D3DXFLOAT16(+1), D3DXFLOAT16(0),
  D3DXFLOAT16(+1), D3DXFLOAT16(-1), D3DXFLOAT16(-3), D3DXFLOAT16(0),
  D3DXFLOAT16(+2), D3DXFLOAT16(-3), D3DXFLOAT16(+3), D3DXFLOAT16(0),
  D3DXFLOAT16(-2), D3DXFLOAT16(-5), D3DXFLOAT16(-3), D3DXFLOAT16(0),
};

void CubeShadowMapPointLightBatch::Draw(LightBuffer* pLightBuffer,
                                        const StructuredBuffer* pCullDataBuffer,
                                        RenderTarget2D* pNormalBuffer, RenderTarget2D* pDepthRT,
                                        RenderTarget2D* pGeomNormalBuffer, Camera* pCamera, 
                                        bool bBlending, DeviceContext11& dc)
{
  LightingShaderFlags lsf;
  lsf.BLENDING = bBlending;
  CubeShadowMapPointLightBatch::Draw(lsf, pLightBuffer, pCullDataBuffer, pNormalBuffer, pDepthRT, pGeomNormalBuffer, pCamera, dc);
}

void CubeShadowMapPointLightBatch::Draw(const LightingShaderFlags& lsf, LightBuffer* pLightBuffer,
                                        const StructuredBuffer* pCullDataBuffer,
                                        RenderTarget2D* pNormalBuffer, RenderTarget2D* pDepthRT,
                                        RenderTarget2D* pGeomNormalBuffer, Camera* pCamera,
                                        DeviceContext11& dc)
{
  dc.PushRC();
  dc.UnbindRT(0);
  dc.UnbindDepthStencil();

  size_t NLights = std::min(m_ToRender.size(), size_t(MAX_LIGHTS_PER_PASS));
  if(m_ToRender.size()!=NLights)
    m_ToRender.resize(NLights);

  PointLightShaderData* pPointLightData = (PointLightShaderData*)Platform::AllocateScratchMemory(sizeof(PointLightShaderData)*NLights);
  Vec4* pShadowMapData = (Vec4*)Platform::AllocateScratchMemory(sizeof(Vec4)*NLights);

  Light** ppLight = &m_ToRender.front();
  bool perQuadShadows = false;
  for(size_t i=0; i<NLights; ++i, ++ppLight)
  {
    CubeShadowMapPointLight* pLight = static_cast<CubeShadowMapPointLight*>(*ppLight);
    pPointLightData[i] = pLight->m_ShaderData;
    pShadowMapData[i] = pLight->m_SMapShaderData;
    perQuadShadows |= pLight->m_FastRender;
  }
  LargeConstantBuffer lightDataCB(sizeof(*pPointLightData)*NLights, pPointLightData, dc);
  LargeConstantBuffer shadowMapDataCB(sizeof(*pShadowMapData)*NLights, pShadowMapData, dc);
  dc.CSSetConstantBuffer(PointLightsShaderData::BANK, lightDataCB.GetBuffer());
  dc.CSSetConstantBuffer(2, shadowMapDataCB.GetBuffer());

  Platform::FreeScratchMemory(pPointLightData);
  Platform::FreeScratchMemory(pShadowMapData);

  dc.BindCS(3, pGeomNormalBuffer);
  static StaticTexture2D<1, 1, DXGI_FORMAT_R16G16B16A16_FLOAT, 1, D3D11_USAGE_IMMUTABLE,
    D3D11_BIND_SHADER_RESOURCE, 0, D3D11_RESOURCE_MISC_TEXTURECUBE, 6> s_IndexTexture(c_CubeIndexData);
  dc.BindCS(6, pDepthRT);

  bool useCBMArray = m_CBMArray.GetDesc().ArraySize>0;
  if(useCBMArray)
  {
    dc.BindCS(4, &m_CBMArray);
    dc.SetSamplerCS(0, &Platform::GetSamplerCache().GetByIndex(Platform::Sampler_ShadowMap_PCF));
  }
  else
  {
    dc.BindCS(4, &s_IndexTexture);
    dc.BindCS(5, &m_DepthAtlas);
    dc.SetSamplerCS(0, &Platform::GetSamplerCache().GetByIndex(Platform::Sampler_Point_Clamp));
    dc.SetSamplerCS(1, &Platform::GetSamplerCache().GetByIndex(Platform::Sampler_ShadowMap_PCF));
  }

  LightingQuadShaderFlags lqf;
  lqf.SHADOWS_CUBEMAP = perQuadShadows;
  lqf.USE_CBM_ARRAY = useCBMArray;
  lqf.TGSM_WORKAROUND = Platform::GetFeatureLevel()<D3D_FEATURE_LEVEL_11_0; // temp fix for NVIDIA driver problem
  lqf.HEMISPHERICAL_LIGHT = lsf.HEMISPHERICAL_LIGHT;
  GetShader(lqf).Bind(dc);

  const float c_SelfShadowOffset = 1.6f;
  Vec4 atlasData(0.5f*(float)m_FaceSize/(float)m_DepthAtlas.GetDesc().Width,
                 0.5f*(float)m_FaceSize/(float)m_DepthAtlas.GetDesc().Height,
                 (float)(m_FaceSize - (m_PCFKernelSize<<1))/(float)m_FaceSize,
                 c_SelfShadowOffset/(float)m_FaceSize);
  PreallocatedPushBuffer<> pb;
  pb.PushConstantCS(NULL, 0, 0);
  if(perQuadShadows)
  {
    pb.PushConstantCS(GetInvViewportTransform(pCamera, pLightBuffer->m_Width, pLightBuffer->m_Height));
    pb.PushConstantCS(atlasData);
  }
  if(lqf.HEMISPHERICAL_LIGHT)
  {
    Mat4x4 A = Mat4x4::ScalingTranslationD3D(Vec3(2.0f/float(pLightBuffer->m_Width), -2.0f/float(pLightBuffer->m_Height), 1), Vec3(-1.0f, 1.0f, 0))*pCamera->GetProjectionInverse();
    Vec3 p00 = Vec3::Project(Vec3(0, 0, 0), A);
    Vec3 p01 = Vec3::Project(Vec3(0, 1, 0), A);
    Vec3 p10 = Vec3::Project(Vec3(1, 0, 0), A);
    float rNear = 1.0f/pCamera->GetNear();

    pb.PushConstantCS(Mat4x4::Transpose(pCamera->GetViewMatrixInverse()));
    pb.PushConstantCS(Vec3::Point(p00*rNear));
    pb.PushConstantCS((p10.x - p00.x)*rNear);
    pb.PushConstantCS((p01.y - p00.y)*rNear);
    pb.PushConstantCS(sqrtf(Vec3::LengthSq(p10 - p00) + Vec3::LengthSq(p01 - p00))*LIGHTING_QUAD_SIZE*rNear);
    pb.PushConstantCS(0.0f);
  }
  pLightBuffer->RunCulling(pb, pCullDataBuffer, dc);

  dc.UnbindCS(6);

  LightingShaderFlags _lsf(lsf);
  _lsf.SHADOWS_CUBEMAP = 1;
  _lsf.FAST_RENDER = perQuadShadows;
  _lsf.USE_PCF9 = (m_PCFKernelSize>1) & !perQuadShadows;
  _lsf.USE_CBM_ARRAY = useCBMArray;
  GetShader(_lsf).Bind(dc);

  if(_lsf.USE_PCF9)
    atlasData.w *= 2.0f;

  pb.Clear();
  pb.PushConstantCS(atlasData, 0);
  pLightBuffer->RunLighting(pb, pNormalBuffer, pDepthRT, pGeomNormalBuffer, pCamera, dc);

  dc.PopRC();
}

const Mat4x4 CubeShadowMapPointLightBatch::GetProjectionMatrix(float range) const
{
  const float c_Near = 0.05f;
  float e = c_Near;
  if(!m_CBMArray.GetDesc().ArraySize)
    e += (float)(m_PCFKernelSize<<1)*c_Near/(float)(m_FaceSize - (m_PCFKernelSize<<1));
  return Mat4x4::ProjectionD3D(-e, e, -e, e, c_Near, range);
}

static const Mat4x4 c_CubeFaceMat[6] =
{
  Mat4x4::Transpose(Mat4x4(-c_ZAxis, c_YAxis, c_XAxis, c_WAxis)),
  Mat4x4::Transpose(Mat4x4( c_ZAxis, c_YAxis,-c_XAxis, c_WAxis)),
  Mat4x4::Transpose(Mat4x4( c_XAxis,-c_ZAxis, c_YAxis, c_WAxis)),
  Mat4x4::Transpose(Mat4x4( c_XAxis, c_ZAxis,-c_YAxis, c_WAxis)),
  Mat4x4::Transpose(Mat4x4( c_XAxis, c_YAxis, c_ZAxis, c_WAxis)),
  Mat4x4::Transpose(Mat4x4(-c_XAxis, c_YAxis,-c_ZAxis, c_WAxis)),
};

const Mat4x4 CubeShadowMapPointLight::GetViewMatrix(int faceIndex) const
{
  return Mat4x4::TranslationD3D(-GetPosition())*c_CubeFaceMat[faceIndex];
}

static void ClearDepth(const Vec4& rect, DeviceContext11& dc)
{
  D3D11_VIEWPORT vp = { rect.x, rect.y, rect.z, rect.w, 0, 1.0f };
  dc.SetViewport(vp);

  static const D3D11_INPUT_ELEMENT_DESC c_InputDesc = {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0};
  static size_t s_ShaderIndex = g_SimpleShaderCache.GetIndex(SimpleShaderDesc("_Shaders\\ClearDepth.shader", NULL, "_Shaders\\ClearDepth.shader", NULL, NULL, NULL, &c_InputDesc, 1));
  g_SimpleShaderCache.GetByIndex(s_ShaderIndex).Bind(dc);

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

  dc.FlushToDevice()->Draw(4, 0);

  dc.UnsetDepthStencilState();
}

void CubeShadowMapPointLightBatch::Update(SceneRenderer* pRenderer, Scene* pScene, CubeShadowMapPointLight* pLight, DeviceContext11& dc)
{
  _ASSERT(pLight->m_Batch==this);

  dc.PushRC();
  dc.UnbindRT(0);

  float lightRange = pLight->GetRange();
  Mat4x4 projMat = GetProjectionMatrix(lightRange);

  CubeMapGlobalConst gc;
  for(int i=0; i<6; ++i)
    gc.g_CubeMapViewProjection[i] = pLight->GetViewMatrix(i)*projMat;
  ID3D11Buffer* pBuffer = dc.GetConstantBuffers().Allocate(sizeof(gc), &gc, dc.DoNotFlushToDevice());
  dc.GSSetConstantBuffer(CubeMapGlobalConst::BANK, pBuffer);

  Camera camera;
  camera.SetViewMatrix(Mat4x4::TranslationD3D(-pLight->GetPosition()));
  camera.SetProjection(Mat4x4::OrthoD3D(-lightRange, lightRange, -lightRange, lightRange, -lightRange, lightRange));
  if(pLight->m_ArraySlice<0)
  {
    pLight->m_SMapShaderData.x = projMat.e43;
    pLight->m_SMapShaderData.y = projMat.e33;
    pLight->m_SMapShaderData.z = (float)pLight->m_SMapRect.x/(float)m_DepthAtlas.GetDesc().Width;
    pLight->m_SMapShaderData.w = (float)pLight->m_SMapRect.y/(float)m_DepthAtlas.GetDesc().Height;

    dc.BindDepthStencil(&m_DepthAtlas);

    ClearDepth(Vec4i::Convert(pLight->m_SMapRect), dc);

    dc.UnbindViewport();
    D3D11_VIEWPORT viewports[6] = { };
    for(int i=0; i<6; ++i)
    {
      viewports[i].TopLeftX = float((i%3)*m_FaceSize + pLight->m_SMapRect.x);
      viewports[i].TopLeftY = float((i/3)*m_FaceSize + pLight->m_SMapRect.y);
      viewports[i].Width = viewports[i].Height = float(m_FaceSize);
      viewports[i].MaxDepth = 1.0f;
    }
    dc.RSSetViewports(ARRAYSIZE(viewports), viewports);

    pRenderer->DrawCubeShadowMap(pScene, &camera, 0);
  }
  else
  {
    pLight->m_SMapShaderData.x = 1.0f/(lightRange*lightRange);
    pLight->m_SMapShaderData.y = (float)(pLight->m_ArraySlice/6);
    pLight->m_SMapShaderData.z = 0;
    pLight->m_SMapShaderData.w = 0;

    dc.BindDepthStencil(&m_CBMArray);
    m_CBMArray.SetViewport(dc);

    for(int i=0; i<6; ++i)
      dc.DoNotFlushToDevice()->ClearDepthStencilView(m_CBMArray.GetDepthStencilView(pLight->m_ArraySlice + i)->GetDepthStencilView(), D3D11_CLEAR_DEPTH, 1.0f, 0);

    CubeMapsArrayConfig acfg;
    acfg.g_FirstSliceIndex = pLight->m_ArraySlice;
    acfg.g_InvViewRange = 1.0f/lightRange;
    ID3D11Buffer* pCubeMapsArrayConfigBuffer = dc.GetConstantBuffers().Allocate(sizeof(acfg), &acfg, dc.DoNotFlushToDevice());
    dc.VSSetConstantBuffer(CubeMapsArrayConfig::BANK, pCubeMapsArrayConfigBuffer);
    dc.GSSetConstantBuffer(CubeMapsArrayConfig::BANK, pCubeMapsArrayConfigBuffer);

    pRenderer->DrawCubeShadowMapArray(pScene, &camera, 0);

    dc.GetConstantBuffers().Free(pCubeMapsArrayConfigBuffer);
  }
  dc.GetConstantBuffers().Free(pBuffer);
  dc.PopRC();
}

////////////////////////////////////////////////////////////////////////

HemisphericalPointLight::HemisphericalPointLight(SceneObject* pParent, SceneQTreeNode* pQTreeRoot) : 
  HemisphericalLight(pParent, pQTreeRoot)
{
  m_Batch = g_HemisphericalPointLightBatch;
}

void HemisphericalPointLightBatch::Draw(LightBuffer* pLightBuffer,
                                        const StructuredBuffer* pCullDataBuffer,
                                        RenderTarget2D* pNormalBuffer, RenderTarget2D* pDepthRT,
                                        RenderTarget2D* pGeomNormalBuffer, Camera* pCamera, 
                                        bool bBlending, DeviceContext11& dc)
{
  size_t NLights = std::min(m_ToRender.size(), size_t(MAX_LIGHTS_PER_PASS));
  if(m_ToRender.size()!=NLights)
    m_ToRender.resize(NLights);

  Vec4* pShaderData = (Vec4*)Platform::AllocateScratchMemory(sizeof(Vec4)*NLights);
  Light** ppLight = &m_ToRender.front();
  for(size_t i=0; i<NLights; ++i, ++ppLight)
    pShaderData[i] = static_cast<HemisphericalPointLight*>(*ppLight)->m_HLightShaderData;

  LargeConstantBuffer shaderDataCB(sizeof(*pShaderData)*m_ToRender.size(), pShaderData, dc);
  dc.CSSetConstantBuffer(HemisphericalLightData::BANK, shaderDataCB.GetBuffer());
  Platform::FreeScratchMemory(pShaderData);

  LightingShaderFlags lsf;
  lsf.BLENDING = bBlending;
  lsf.HEMISPHERICAL_LIGHT = 1;
  PointLightBatch::Draw(lsf, pLightBuffer, pCullDataBuffer, pNormalBuffer, pDepthRT, pGeomNormalBuffer, pCamera, dc);
}

////////////////////////////////////////////////////////////////////////

HemisphericalCubeShadowMapPointLight::HemisphericalCubeShadowMapPointLight(unsigned faceSize, SceneObject* pParent, SceneQTreeNode* pQTreeRoot, bool fastRender) :
  HemisphericalLight(pParent, pQTreeRoot)
{
  m_FastRender = fastRender;
  HemisphericalCubeShadowMapPointLightBatch::Allocate(this, faceSize);
  CubeShadowMapPointLight::s_Lights.insert(this);
}

void HemisphericalCubeShadowMapPointLightBatch::Allocate(HemisphericalCubeShadowMapPointLight* pLight, unsigned faceSize)
{
  auto it = std::find_if(s_Batches.cbegin(), s_Batches.cend(), 
    [&] (CubeShadowMapPointLightBatch* b) -> bool { return b->m_FaceSize==faceSize && std::max(b->m_FreeRect.size(), b->m_FreeSlice.size())>0; });
  (it!=s_Batches.cend() ? *it : new HemisphericalCubeShadowMapPointLightBatch(faceSize))->Add(pLight);
}

void HemisphericalCubeShadowMapPointLightBatch::Draw(LightBuffer* pLightBuffer,
                                                     const StructuredBuffer* pCullDataBuffer,
                                                     RenderTarget2D* pNormalBuffer, RenderTarget2D* pDepthRT,
                                                     RenderTarget2D* pGeomNormalBuffer, Camera* pCamera, 
                                                     bool bBlending, DeviceContext11& dc)
{
  size_t NLights = std::min(m_ToRender.size(), size_t(MAX_LIGHTS_PER_PASS));
  if(m_ToRender.size()!=NLights)
    m_ToRender.resize(NLights);

  Vec4* pShaderData = (Vec4*)Platform::AllocateScratchMemory(sizeof(Vec4)*NLights);
  Light** ppLight = &m_ToRender.front();
  for(size_t i=0; i<NLights; ++i, ++ppLight)
    pShaderData[i] = static_cast<HemisphericalCubeShadowMapPointLight*>(*ppLight)->m_HLightShaderData;

  LargeConstantBuffer shaderDataCB(sizeof(*pShaderData)*m_ToRender.size(), pShaderData, dc);
  dc.CSSetConstantBuffer(HemisphericalLightData::BANK, shaderDataCB.GetBuffer());
  Platform::FreeScratchMemory(pShaderData);

  LightingShaderFlags lsf;
  lsf.BLENDING = bBlending;
  lsf.HEMISPHERICAL_LIGHT = 1;
  CubeShadowMapPointLightBatch::Draw(lsf, pLightBuffer, pCullDataBuffer, pNormalBuffer, pDepthRT, pGeomNormalBuffer, pCamera, dc);
}
