#include "PreCompile.h"
#include "RapidXML/rapidxml.hpp"
#include "delegate/Delegate.h"
#include "MeshInstance.h"
#include "Attribute.h"
#include "TextureLoader/TextureLoader.h"
#include "Platform11/IABuffer11.h"
#include "Shaders.h"
#include "../../Math/IO.h"
#include "../../Util/Log.h"

IMPLEMENT_MEMORY_POOL(MeshInstance, 4096);

class MaterialDesc
{
public:
  enum
  {
    FirstTextureType = 0,
    File_DiffuseMap = FirstTextureType,
    File_NormalMap,
    File_GlossMap,
    File_DetailMap,
    LastTextureType,
    File_Geometry = LastTextureType,
    NFileTypes,
    NFileTextures = LastTextureType - FirstTextureType,
  };
  enum MaterialType
  {
    Type_Default = 0,
    Type_DepthOnly,
  };

  void Clear()
  {
    for(int i=0; i<NFileTypes; ++i) m_FileName[i].clear();
    memset(&m_PrePassShaderFlags, 0, sizeof(m_PrePassShaderFlags));
    m_bConvertToDepthOnly = true;
    m_Type = Type_Default;
  }

  template<unsigned c_FileType> void SetFileName(const char* str) { m_FileName[c_FileType] = str; }
  void SetALPHATESTED(const char* str) { m_PrePassShaderFlags.ALPHATESTED = MathIO::ReadInteger(str); }
  void SetVERTEXCOLOR(const char* str) { m_PrePassShaderFlags.VERTEXCOLOR = MathIO::ReadInteger(str); }
  void SetBILLBOARD(const char* str) { m_PrePassShaderFlags.BILLBOARD = MathIO::ReadInteger(str); }
  void SetSPECULAR(const char* str) { m_PrePassShaderFlags.SPECULAR = MathIO::ReadInteger(str); }
  void SetType(const char* str) { m_Type = (MaterialType)MathIO::ReadInteger(str); }
  void SetConvertToDepthOnly(const char* str) { m_bConvertToDepthOnly = MathIO::ReadInteger(str)!=0; }

  const std::string& GetFileName(int fileType) const { return m_FileName[fileType]; }
  const PrePassShaderFlags& GetPrePassShaderFlags() const { return m_PrePassShaderFlags; }
  const MaterialType GetType() const { return m_Type; }
  const bool ConvertToDepthOnly() const { return m_bConvertToDepthOnly; }

protected:
  std::string m_FileName[NFileTypes];
  PrePassShaderFlags m_PrePassShaderFlags;
  bool m_bConvertToDepthOnly;
  MaterialType m_Type;
};

bool Mesh::LoadXML(const char* pszFileName, TextureLoader& texLoader)
{
  char meshFullPath[256];
  char buf[4096];
  MemoryBuffer file(sizeof(buf), buf);
  if(!file.Load(Platform::GetPath(Platform::File_Mesh, meshFullPath, pszFileName)))
  {
    Log::Error("failed to load \"%s\"\n", pszFileName);
    return false;
  }
  file.Seek(file.Size());
  file.Write((char)0);
  rapidxml::xml_document<> doc;
  try
  {
    doc.parse<0>(file.Ptr<char>(0));
  }
  catch(rapidxml::parse_error& e)
  {
    Log::Error("error parsing \"%s\": %s\n", pszFileName, e.what());
    return false;
  }

  static MaterialDesc s_MatDesc;
  static Attribute s_AttrData[] =
  {
    Attribute("Geometry", Attribute::Setter::from_method<MaterialDesc, &MaterialDesc::SetFileName<MaterialDesc::File_Geometry> >(&s_MatDesc)),
    Attribute("DiffuseMap", Attribute::Setter::from_method<MaterialDesc, &MaterialDesc::SetFileName<MaterialDesc::File_DiffuseMap> >(&s_MatDesc)),
    Attribute("NormalMap", Attribute::Setter::from_method<MaterialDesc, &MaterialDesc::SetFileName<MaterialDesc::File_NormalMap> >(&s_MatDesc)),
    Attribute("GlossMap", Attribute::Setter::from_method<MaterialDesc, &MaterialDesc::SetFileName<MaterialDesc::File_GlossMap> >(&s_MatDesc)),
    Attribute("DetailMap", Attribute::Setter::from_method<MaterialDesc, &MaterialDesc::SetFileName<MaterialDesc::File_DetailMap> >(&s_MatDesc)),
    Attribute("ALPHATESTED", Attribute::Setter::from_method<MaterialDesc, &MaterialDesc::SetALPHATESTED>(&s_MatDesc)),
    Attribute("VERTEXCOLOR", Attribute::Setter::from_method<MaterialDesc, &MaterialDesc::SetVERTEXCOLOR>(&s_MatDesc)),
    Attribute("BILLBOARD", Attribute::Setter::from_method<MaterialDesc, &MaterialDesc::SetBILLBOARD>(&s_MatDesc)),
    Attribute("SPECULAR", Attribute::Setter::from_method<MaterialDesc, &MaterialDesc::SetSPECULAR>(&s_MatDesc)),
    Attribute("Type", Attribute::Setter::from_method<MaterialDesc, &MaterialDesc::SetType>(&s_MatDesc)),
    Attribute("ConvertToDepthOnly", Attribute::Setter::from_method<MaterialDesc, &MaterialDesc::SetConvertToDepthOnly>(&s_MatDesc)),
  };
  static const Attributes s_Attributes(ARRAYSIZE(s_AttrData), s_AttrData);

  rapidxml::xml_node<>* pNode = doc.first_node();
  while(pNode!=NULL)
  {
    if(!strcmp(pNode->name(), "Material"))
    {
      s_MatDesc.Clear();
      rapidxml::xml_attribute<>* pAttr = pNode->first_attribute();
      while(pAttr!=NULL)
      {
        Attribute* p = s_Attributes.Find(pAttr->name());
        if(p!=NULL) p->Set(pAttr->value());
        pAttr = pAttr->next_attribute();
      }
      if(!AddMaterial(s_MatDesc, texLoader))
        return false;
    }
    else if(!strcmp(pNode->name(), "AABB"))
    {
      rapidxml::xml_attribute<>* pMin = pNode->first_attribute("Min"); _ASSERT(pMin!=NULL);
      rapidxml::xml_attribute<>* pMax = pNode->first_attribute("Max"); _ASSERT(pMax!=NULL);
      Vec3 AABBMin = pMin!=NULL ? MathIO::Read<Vec3, 3>(pMin->value()) : Vec3::Zero();
      Vec3 AABBMax = pMax!=NULL ? MathIO::Read<Vec3, 3>(pMax->value()) : Vec3::Zero();
      m_AABB = Mat4x4::ScalingTranslationD3D(0.5f*(AABBMax - AABBMin), 0.5f*(AABBMax + AABBMin));
    }
    pNode = pNode->next_sibling();
  }

  m_Info = pszFileName;
  return true;
}

Mesh::Mesh() : m_AABB(Mat4x4::Identity())
{
  m_Textures.reserve(12);
  m_Buffers.reserve(8);
  m_ToRender.reserve(8);
}

Mesh::~Mesh()
{
  for(size_t i=0; i<m_Textures.size(); ++i)
    m_Textures[i]->Destruct();
  for(size_t i=0; i<m_Buffers.size(); ++i)
    m_Buffers[i]->Destruct();
}

inline unsigned GetSlot(unsigned fileType)
{
  return fileType;
}

bool Mesh::AddMaterial(const MaterialDesc& matDesc, TextureLoader& texLoader)
{
  const std::string& geometryFileName = matDesc.GetFileName(MaterialDesc::File_Geometry);
  char geometryFullPath[256];
  MemoryBuffer geom;
  if(!geom.Load(Platform::GetPath(Platform::File_Mesh, geometryFullPath, geometryFileName.c_str())))
  {
    Log::Error("failed to load \"%s\"\n", geometryFileName.c_str());
    return false;
  }
  VertexFormatDesc vertexDesc;
  vertexDesc.Deserialize(geom);

  IABuffer* pVB = new IABuffer();
  m_Buffers.push_back(pVB);
  unsigned nVertices = geom.Read<unsigned>();
  unsigned vertexSize = vertexDesc.GetMinVertexSize();
  HRESULT hr = pVB->Init(nVertices, vertexSize, geom.Ptr<void>());
  if(FAILED(hr))
  {
    Log::Error("failed to create VB from \"%s\" (HRESULT=0x%x)\n", geometryFileName.c_str(), hr);
    return false;
  }
  geom.Seek(geom.Position() + nVertices*vertexSize);

  IABuffer* pIB = new IABuffer();
  m_Buffers.push_back(pIB);
  unsigned nIndices = geom.Read<unsigned>();
  const unsigned indexSize = sizeof(unsigned short);
  hr = pIB->Init(nIndices, indexSize, geom.Ptr<void>(), D3D11_USAGE_IMMUTABLE, D3D11_BIND_INDEX_BUFFER);
  if(FAILED(hr))
  {
    Log::Error("failed to create IB from \"%s\" (HRESULT=0x%x)\n", geometryFileName.c_str(), hr);
    return false;
  }
  geom.Seek(geom.Position() + nIndices*indexSize);

  Texture2D* pTextures[MaterialDesc::NFileTypes] = { };
  for(int i=MaterialDesc::FirstTextureType; i<MaterialDesc::LastTextureType; ++i)
  {
    const std::string& textureFileName = matDesc.GetFileName(i);
    if(!textureFileName.empty())
    {
      pTextures[i] = texLoader.Get(FileTextureDesc(textureFileName.c_str())).Clone();
      if(pTextures[i]!=NULL)
        m_Textures.push_back(pTextures[i]);
    }
  }

  PrePassShaderFlags prepassShaderFlags = matDesc.GetPrePassShaderFlags();
  prepassShaderFlags.DIFFUSEMAP = (pTextures[MaterialDesc::File_DiffuseMap]!=NULL);
  prepassShaderFlags.NORMALMAP = (pTextures[MaterialDesc::File_NormalMap]!=NULL);
  prepassShaderFlags.GLOSSMAP = (pTextures[MaterialDesc::File_GlossMap]!=NULL);
  prepassShaderFlags.DETAILMAP = (pTextures[MaterialDesc::File_DetailMap]!=NULL);

  RenderContext11 rc;
  rc.SetRasterizerState(&Platform::GetRasterizerCache().ConcurrentGet(RasterizerDesc11()));
  rc.SetBlendState(&Platform::GetBlendCache().ConcurrentGet(BlendDesc11()));
  rc.SetDepthStencilState(&Platform::GetDepthStencilCache().ConcurrentGet(DepthStencilDesc11()));
  rc.BindVertexBuffer(0, pVB, 0);
  rc.BindIndexBuffer(pIB, 0);
  rc.SetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

  static const SamplerState11* s_Samplers[MaterialDesc::NFileTextures] =
  {
    &Platform::GetSamplerCache().GetByIndex(Platform::Sampler_Linear_Wrap),
    &Platform::GetSamplerCache().GetByIndex(Platform::Sampler_Linear_Wrap),
    &Platform::GetSamplerCache().GetByIndex(Platform::Sampler_Linear_Wrap),
    &Platform::GetSamplerCache().GetByIndex(Platform::Sampler_Linear_Wrap),
  };

  if(matDesc.GetType()==MaterialDesc::Type_DepthOnly ||
    (matDesc.GetType()==MaterialDesc::Type_Default && matDesc.ConvertToDepthOnly()))
  {
    DepthOnlyShaderFlags depthOnlyShaderFlags;
    depthOnlyShaderFlags.ALPHATESTED = prepassShaderFlags.DIFFUSEMAP & prepassShaderFlags.ALPHATESTED;
    depthOnlyShaderFlags.BILLBOARD = prepassShaderFlags.BILLBOARD;

    if(depthOnlyShaderFlags.ALPHATESTED)
    {
      const unsigned Slot = GetSlot(MaterialDesc::File_DiffuseMap);
      rc.BindPS(Slot, pTextures[MaterialDesc::File_DiffuseMap]);
      rc.SetSamplerPS(Slot, s_Samplers[MaterialDesc::File_DiffuseMap]);
    }

    static DepthOnlyShaderCache s_DepthOnlyShaderCache(256);

    s_DepthOnlyShaderCache.Get(DepthOnlyShaderDesc(depthOnlyShaderFlags, vertexDesc)).Bind(rc);

    if(m_ShadowMapPB.IsEmpty())
      m_ShadowMapUpdateHandles.hTransform = m_ShadowMapPB.PushConstantVS(NULL, sizeof(Mat4x4)*c_NInstancesMax, 0);
    m_ShadowMapPB.PushRC(rc);
    m_ShadowMapPB.DrawIndexedInstancedVariant(nIndices, 0, 0, 0);

    depthOnlyShaderFlags.CUBEMAP = 1;
    s_DepthOnlyShaderCache.Get(DepthOnlyShaderDesc(depthOnlyShaderFlags, vertexDesc)).Bind(rc);

    if(m_CubeShadowMapPB.IsEmpty())
      m_CubeShadowMapUpdateHandles.hTransform = m_CubeShadowMapPB.PushConstantVS(NULL, sizeof(Mat4x4)*c_NInstancesMax, 0);
    m_CubeShadowMapPB.PushRC(rc);
    m_CubeShadowMapPB.DrawIndexedInstancedVariant(nIndices, 0, 0, 0);

    depthOnlyShaderFlags.CUBEMAPS_ARRAY = 1;
    s_DepthOnlyShaderCache.Get(DepthOnlyShaderDesc(depthOnlyShaderFlags, vertexDesc)).Bind(rc);

    if(m_CubeShadowMapArrayPB.IsEmpty())
      m_CubeShadowMapArrayUpdateHandles.hTransform = m_CubeShadowMapArrayPB.PushConstantVS(NULL, sizeof(Mat4x4)*c_NInstancesMax, 0);
    m_CubeShadowMapArrayPB.PushRC(rc);
    m_CubeShadowMapArrayPB.DrawIndexedInstancedVariant(nIndices, 0, 0, 0);

    depthOnlyShaderFlags.CUBEMAP = 0;
    depthOnlyShaderFlags.CUBEMAPS_ARRAY = 0;
    depthOnlyShaderFlags.ASM_LAYER = 1;
    s_DepthOnlyShaderCache.Get(DepthOnlyShaderDesc(depthOnlyShaderFlags, vertexDesc)).Bind(rc);

    if(m_ASMLayerShadowMapPB.IsEmpty())
      m_ASMLayerShadowMapUpdateHandles.hTransform = m_ASMLayerShadowMapPB.PushConstantVS(NULL, sizeof(Mat4x4)*c_NInstancesMax, 0);
    m_ASMLayerShadowMapPB.PushRC(rc);
    m_ASMLayerShadowMapPB.DrawIndexedInstancedVariant(nIndices, 0, 0, 0);

    depthOnlyShaderFlags.ASM_LAYER = 0;
    depthOnlyShaderFlags.PARABOLIC = 1;
    s_DepthOnlyShaderCache.Get(DepthOnlyShaderDesc(depthOnlyShaderFlags, vertexDesc)).Bind(rc);

    rc.SetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST);
    rc.SetRasterizerState(&Platform::GetRasterizerCache().ConcurrentGet(RasterizerDesc11(D3D11_FILL_WIREFRAME)));

    if(m_ParabolicShadowMapPB.IsEmpty())
      m_ParabolicShadowMapUpdateHandles.hTransform = m_ParabolicShadowMapPB.PushConstantVS(NULL, sizeof(Mat4x4)*c_NInstancesMax, 0);
    m_ParabolicShadowMapPB.PushRC(rc);
    m_ParabolicShadowMapPB.DrawIndexedInstancedVariant(nIndices, 0, 0, 0);

    rc.SetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    rc.SetRasterizerState(&Platform::GetRasterizerCache().ConcurrentGet(RasterizerDesc11()));
  }

  if(matDesc.GetType()==MaterialDesc::Type_Default)
  {
    static PrePassShaderCache s_PrePassShaderCache(512);
    s_PrePassShaderCache.Get(PrePassShaderDesc(prepassShaderFlags, vertexDesc)).Bind(rc);

    for(int i=MaterialDesc::FirstTextureType; i<MaterialDesc::LastTextureType; ++i)
    {
      if(pTextures[i]==NULL) continue;
      const unsigned Slot = GetSlot(i);
      rc.BindPS(Slot, pTextures[i]);
      rc.SetSamplerPS(Slot, s_Samplers[i]);
    }

    if(m_PrePassPB.IsEmpty())
      m_PrePassUpdateHandles.hTransform = m_PrePassPB.PushConstantVS(NULL, sizeof(Mat4x4)*c_NInstancesMax, 0);
    m_PrePassPB.PushRC(rc);
    m_PrePassPB.DrawIndexedInstancedVariant(nIndices, 0, 0, 0);
  }

  return true;
}

MeshInstance::MeshInstance(Mesh* pMesh, SceneObject* pParent, SceneQTreeNode* pQTreeRoot) : 
  m_Mesh(pMesh), SceneObject(pMesh->m_AABB, pParent, pQTreeRoot),
  m_PrePassUpdateHandles(m_Mesh->m_PrePassUpdateHandles),
  m_ShadowMapUpdateHandles(m_Mesh->m_ShadowMapUpdateHandles),
  m_CubeShadowMapUpdateHandles(m_Mesh->m_CubeShadowMapUpdateHandles),
  m_CubeShadowMapArrayUpdateHandles(m_Mesh->m_CubeShadowMapArrayUpdateHandles),
  m_ParabolicShadowMapUpdateHandles(m_Mesh->m_ParabolicShadowMapUpdateHandles),
  m_ASMLayerShadowMapUpdateHandles(m_Mesh->m_ASMLayerShadowMapUpdateHandles)
{
  m_PrePassPB.Append(m_Mesh->m_PrePassPB, sizeof(m_PrePassUpdateHandles)/sizeof(PushBufferConstantHandle), (PushBufferConstantHandle*)&m_PrePassUpdateHandles);
  m_ShadowMapPB.Append(m_Mesh->m_ShadowMapPB, sizeof(m_ShadowMapUpdateHandles)/sizeof(PushBufferConstantHandle), (PushBufferConstantHandle*)&m_ShadowMapUpdateHandles);
  m_CubeShadowMapPB.Append(m_Mesh->m_CubeShadowMapPB, sizeof(m_CubeShadowMapUpdateHandles)/sizeof(PushBufferConstantHandle), (PushBufferConstantHandle*)&m_CubeShadowMapUpdateHandles);
  m_CubeShadowMapArrayPB.Append(m_Mesh->m_CubeShadowMapArrayPB, sizeof(m_CubeShadowMapArrayUpdateHandles)/sizeof(PushBufferConstantHandle), (PushBufferConstantHandle*)&m_CubeShadowMapArrayUpdateHandles);
  m_ParabolicShadowMapPB.Append(m_Mesh->m_ParabolicShadowMapPB, sizeof(m_ParabolicShadowMapUpdateHandles)/sizeof(PushBufferConstantHandle), (PushBufferConstantHandle*)&m_ParabolicShadowMapUpdateHandles);
  m_ASMLayerShadowMapPB.Append(m_Mesh->m_ASMLayerShadowMapPB, sizeof(m_ASMLayerShadowMapUpdateHandles)/sizeof(PushBufferConstantHandle), (PushBufferConstantHandle*)&m_ASMLayerShadowMapUpdateHandles);
}

void MeshInstance::OptimizeAsStatic()
{
  m_PrePassPB.CreateConstantBuffers();
  m_ShadowMapPB.CreateConstantBuffers();
  m_CubeShadowMapPB.CreateConstantBuffers();
  m_CubeShadowMapArrayPB.CreateConstantBuffers();
  m_ParabolicShadowMapPB.CreateConstantBuffers();
  m_ASMLayerShadowMapPB.CreateConstantBuffers();
}

void Mesh::DrawInstanced(PushBuffer& pb, PushBufferConstantHandle& hTransform, DeviceContext11& dc)
{
  size_t toRender = m_ToRender.size();
  auto it = m_ToRender.begin();
  while(toRender>0)
  {
    Mat4x4 buf[c_NInstancesMax];
    size_t n = std::min(toRender, c_NInstancesMax);
    for(size_t i=0; i<n; ++i, ++it)
      buf[i] = (*it)->GetTransform();
    pb.Update(hTransform, sizeof(Mat4x4)*n, buf, true);
    pb.Execute(dc, n);
    toRender -= n;
  }
}

void Mesh::DrawPrePass(DeviceContext11& dc)            { Draw<&MeshInstance::GetPrePassPB>(m_PrePassPB, m_PrePassUpdateHandles.hTransform, dc); }
void Mesh::DrawShadowMap(DeviceContext11& dc)          { Draw<&MeshInstance::GetShadowMapPB>(m_ShadowMapPB, m_ShadowMapUpdateHandles.hTransform, dc); }
void Mesh::DrawCubeShadowMap(DeviceContext11& dc)      { Draw<&MeshInstance::GetCubeShadowMapPB>(m_CubeShadowMapPB, m_CubeShadowMapUpdateHandles.hTransform, dc); }
void Mesh::DrawCubeShadowMapArray(DeviceContext11& dc) { Draw<&MeshInstance::GetCubeShadowMapArrayPB>(m_CubeShadowMapArrayPB, m_CubeShadowMapArrayUpdateHandles.hTransform, dc); }
void Mesh::DrawParabolicShadowMap(DeviceContext11& dc) { Draw<&MeshInstance::GetParabolicShadowMapPB>(m_ParabolicShadowMapPB, m_ParabolicShadowMapUpdateHandles.hTransform, dc); }
void Mesh::DrawASMLayerShadowMap(DeviceContext11& dc)  { Draw<&MeshInstance::GetASMLayerShadowMapPB>(m_ASMLayerShadowMapPB, m_ASMLayerShadowMapUpdateHandles.hTransform, dc); }

Renderable* MeshInstance::PrepareToRender()
{
  m_Mesh->m_ToRender.push_back(this);
  return m_Mesh->m_ToRender.size()>1 ? NULL : m_Mesh;
}

void MeshInstance::OnTransformChanged()
{
  __super::OnTransformChanged();
  m_PrePassPB.Update(m_PrePassUpdateHandles.hTransform, m_Transform, true);
  m_ShadowMapPB.Update(m_ShadowMapUpdateHandles.hTransform, m_Transform, true);
  m_CubeShadowMapPB.Update(m_CubeShadowMapUpdateHandles.hTransform, m_Transform, true);
  m_CubeShadowMapArrayPB.Update(m_CubeShadowMapArrayUpdateHandles.hTransform, m_Transform, true);
  m_ParabolicShadowMapPB.Update(m_ParabolicShadowMapUpdateHandles.hTransform, m_Transform, true);
  m_ASMLayerShadowMapPB.Update(m_ASMLayerShadowMapUpdateHandles.hTransform, m_Transform, true);
}
