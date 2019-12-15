#include "PreCompile.h"
#include "RapidXML/rapidxml.hpp"
#include "Scene.h"
#include "Mesh/Attribute.h"
#include "Mesh/MeshInstance.h"
#include "Lighting/Light.h"
#include "../../Util/Frustum.h"
#include "../../Util/Log.h"
#include "../../Math/IO.h"

IMPLEMENT_MEMORY_POOL(SceneQTreeNode, 8192);

SceneObject::SceneObject(const Mat4x4& aabb, SceneObject* pParent, SceneQTreeNode* pQTreeRoot) : 
  m_OBB(aabb), m_Transform(Mat4x4::Identity()), m_Parent(pParent), m_FirstChild(NULL), m_NextSibling(NULL), 
  m_QTreeRoot(pQTreeRoot), m_TimeStamp(0)
{
  if(m_Parent!=NULL)
  {
    m_NextSibling = m_Parent->m_FirstChild;
    m_Parent->m_FirstChild = this;
  }
}

inline void DeleteChildren(SceneObject* pObj, bool bDelete)
{
  while(pObj->GetFirstChild()!=NULL)
    DeleteChildren(pObj->GetFirstChild(), true);
  if(bDelete)
    delete pObj;
}

SceneObject::~SceneObject()
{
  DeleteChildren(this, false);
  if(m_Parent!=NULL)
  {
    if(m_Parent->m_FirstChild!=this)
    {
      SceneObject* pObj;
      for(pObj=m_Parent->m_FirstChild; pObj!=NULL; pObj=pObj->m_NextSibling)
        if(pObj->m_NextSibling==this)
          { pObj->m_NextSibling = m_NextSibling; break; }
      _ASSERT(pObj!=NULL);
    }
    else
      m_Parent->m_FirstChild = m_NextSibling;
  }
  if(m_QTreeRoot!=NULL)
    QTreeNodeObject::Remove();
}

void SceneObject::OnTransformChanged()
{
  if(m_QTreeRoot!=NULL)
  {
    QTreeNodeObject::Remove();
    Vec3 min, max;
    Mat4x4::OBBtoAABB_D3D(m_OBB, min, max);
    Assign(m_QTreeRoot, AABB2D(Vec4::Swizzle<x,z,w,w>(min), Vec4::Swizzle<x,z,w,w>(max)));
  }
}

void SceneObject::UpdateBSphereRadius()
{
  m_BSphereRadius = ::GetBSphereRadius(m_OBB);
}

void SceneObject::SetTransform(const Mat4x4& tm)
{
//  Mat4x4 invTransform = Mat4x4::OBBInverseD3D(m_Transform);
  Mat4x4 invTransform = Mat4x4::Inverse(m_Transform);
  Mat4x4 dm = invTransform*tm;
  Mat4x4 aabb = m_OBB*invTransform;
  m_Transform = tm;
  m_OBB = aabb*tm;
  UpdateBSphereRadius();
  OnTransformChanged();
  for(SceneObject* pObj=m_FirstChild; pObj!=NULL; pObj=pObj->m_NextSibling)
    pObj->SetTransform(pObj->m_Transform*dm);
}

void SceneObject::Assign(SceneQTreeNode* pNode, const AABB2D& objectAABB)
{
  __super::Assign(pNode, objectAABB);
  Vec3 min, max;
  Mat4x4::OBBtoAABB_D3D(m_OBB, min, max);
  for(int i=0; i<4; ++i)
  {
    SceneQTreeNode* pNode = m_BelongsTo[i];
    while(pNode!=NULL && (min.y<pNode->m_MinHeight || max.y>pNode->m_MaxHeight))
    {
      pNode->m_MinHeight = std::min(min.y, pNode->m_MinHeight);
      pNode->m_MaxHeight = std::max(max.y, pNode->m_MaxHeight);
      pNode = pNode->m_Parent;
    }
  }
}

Scene::Scene(const Vec4& AABBMin, const Vec4& AABBMax) : 
  m_QTreeRoot(AABB2D(AABBMin, AABBMax)), m_RootObject(Mat4x4::Identity()), m_TimeStamp(0)
{
}

Scene::~Scene()
{
}

class MeshInstanceDesc : public MathLibObject
{
public:
  void Clear()
  {
    m_Transform = Mat4x4::Identity();
    m_FileName.clear();
  }
  void SetTransform(const char* str)
  {
    float f[12] = { };
    MathIO::ReadTabulatedArray<float, &MathIO::ReadFloat>(str, f, ARRAYSIZE(f));
    m_Transform = Mat4x4::Transpose(Mat4x4(Vec4(&f[0]), Vec4(&f[4]), Vec4(&f[8]), c_WAxis));
  }
  void SetFileName(const char* str) { m_FileName = str; }
  const std::string& GetFileName() const { return m_FileName; }
  const Mat4x4& GetTransform() const { return m_Transform; }

protected:
  Mat4x4 m_Transform;
  std::string m_FileName;
};

class PointLightDesc : public MathLibObject
{
public:
  void Clear()
  {
    m_Position = Vec3::Zero();
    m_Color = Vec4(1.0f);
    m_Range = 1.0f;
    m_ShadowMapResolution = 0;
    m_FastRender = false;
  }
  void SetPosition(const char* str) { m_Position = MathIO::Read<Vec3, 3>(str); }
  void SetColor(const char* str) { m_Color = MathIO::Read<Vec4, 4>(str); }
  void SetRange(const char* str) { m_Range = MathIO::ReadFloat(str); }
  void SetShadowMapResolution(const char* str) { m_ShadowMapResolution = MathIO::ReadInteger(str); }
  void SetFastRender(const char*) { m_FastRender = true; }
  const Vec3& GetPosition() const { return m_Position; }
  const Vec4& GetColor() const { return m_Color; }
  float GetRange() const { return m_Range; }
  unsigned GetShadowMapResolution() const { return m_ShadowMapResolution; }
  bool GetFastRender() const { return m_FastRender; }

protected:
  Vec4 m_Color;
  Vec3 m_Position;
  unsigned m_ShadowMapResolution;
  float m_Range;
  bool m_FastRender;
};

static void ParseAttributes(rapidxml::xml_attribute<>* pAttr, const Attributes& attributes)
{
  while(pAttr!=NULL)
  {
    Attribute* p = attributes.Find(pAttr->name());
    if(p!=NULL) p->Set(pAttr->value());
    pAttr = pAttr->next_attribute();
  }
}

bool Scene::ParseNode(SceneObject* pParent, rapidxml::xml_node<>* pXMLNode)
{
  SceneObject* pObject = NULL;
  if(!strcmp(pXMLNode->name(), "MeshInstance"))
  {
    static MeshInstanceDesc s_Desc;
    static Attribute s_AttrData[] =
    {
      Attribute("Mesh", Attribute::Setter::from_method<MeshInstanceDesc, &MeshInstanceDesc::SetFileName>(&s_Desc)),
      Attribute("Transform", Attribute::Setter::from_method<MeshInstanceDesc, &MeshInstanceDesc::SetTransform>(&s_Desc)),
    };
    static const Attributes s_Attributes(ARRAYSIZE(s_AttrData), s_AttrData);

    s_Desc.Clear();
    ParseAttributes(pXMLNode->first_attribute(), s_Attributes);

    MeshFileDesc fileDesc(s_Desc.GetFileName().c_str());
    Mesh* pMesh = m_Meshes.Get(fileDesc);
    if(pMesh!=NULL)
    {
      MeshInstance* pMeshInstance = new MeshInstance(pMesh, pParent, &m_QTreeRoot);
      pMeshInstance->SetTransform(s_Desc.GetTransform());
      pMeshInstance->OptimizeAsStatic();
      pObject = pMeshInstance;
    }
    else
    {
      Log::Error("error loading mesh \"%s\"\n", s_Desc.GetFileName().c_str());
      return false;
    }
  }
  else if(!strcmp(pXMLNode->name(), "PointLight"))
  {
    static PointLightDesc s_Desc;
    static Attribute s_AttrData[] =
    {
      Attribute("Position", Attribute::Setter::from_method<PointLightDesc, &PointLightDesc::SetPosition>(&s_Desc)),
      Attribute("Color", Attribute::Setter::from_method<PointLightDesc, &PointLightDesc::SetColor>(&s_Desc)),
      Attribute("Range", Attribute::Setter::from_method<PointLightDesc, &PointLightDesc::SetRange>(&s_Desc)),
      Attribute("ShadowMapResolution", Attribute::Setter::from_method<PointLightDesc, &PointLightDesc::SetShadowMapResolution>(&s_Desc)),
      Attribute("FastRender", Attribute::Setter::from_method<PointLightDesc, &PointLightDesc::SetFastRender>(&s_Desc)),
    };
    static const Attributes s_Attributes(ARRAYSIZE(s_AttrData), s_AttrData);

    s_Desc.Clear();
    ParseAttributes(pXMLNode->first_attribute(), s_Attributes);

    PointLight* pLight = s_Desc.GetShadowMapResolution()>0 ?
      new CubeShadowMapPointLight(s_Desc.GetShadowMapResolution(), pParent, &m_QTreeRoot, s_Desc.GetFastRender()) :
      new PointLight(pParent, &m_QTreeRoot);
    pLight->SetColorDeferred(s_Desc.GetColor());
    pLight->SetRangeDeferred(s_Desc.GetRange());
    pLight->SetPositionDeferred(s_Desc.GetPosition());
    pLight->Commit();
    pObject = pLight;
  }
  _ASSERT(pObject!=NULL && "failed to parse node");
  if(pObject!=NULL)
  {
    for(rapidxml::xml_node<>* pChild=pXMLNode->first_node(); pChild!=NULL; pChild=pChild->next_sibling())
      if(ParseNode(pObject, pChild))
        return false;
  }
  return true;
}

bool Scene::LoadXML(const char* pszFileName)
{
  char fullPath[256];
  MemoryBuffer file;
  if(!file.Load(Platform::GetPath(Platform::File_Mesh, fullPath, pszFileName)))
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

  TextureLoader texLoader;
  m_Meshes.Init(&texLoader, false);

  rapidxml::xml_node<>* pNode = doc.first_node();
  while(pNode!=NULL)
  {
    if(!strcmp(pNode->name(), "AABB"))
    {
      rapidxml::xml_attribute<>* pMin = pNode->first_attribute("Min"); _ASSERT(pMin!=NULL);
      rapidxml::xml_attribute<>* pMax = pNode->first_attribute("Max"); _ASSERT(pMax!=NULL);
      Vec2 AABBMin = pMin!=NULL ? MathIO::Read<Vec2, 2>(pMin->value()) : Vec2::Zero();
      Vec2 AABBMax = pMax!=NULL ? MathIO::Read<Vec2, 2>(pMax->value()) : Vec2::Zero();
      _ASSERT(m_RootObject.GetFirstChild()==NULL && "cannot change qtree root");
      m_QTreeRoot = SceneQTreeNode(AABB2D(AABBMin, AABBMax));
    }
    else if(!ParseNode(&m_RootObject, pNode))
    {
      Log::Error("error loading scene \"%s\" at node \"%s\"\n", pszFileName, pNode->name());
      return false;
    }
    pNode = pNode->next_sibling();
  }
  return true;
}

struct SceneInfo : public std::vector<std::pair<size_t, size_t> >
{
  void Get(SceneQTreeNode* pNode, size_t lvl, unsigned timeStamp)
  {
    for(int i=0; i<4; ++i)
      if(pNode->GetChild(i)!=NULL)
        Get(pNode->GetChild(i), lvl + 1, timeStamp);
    resize(std::max(lvl + 1, size()));
    ++at(lvl).first;
    SceneObject* pObj = pNode->GetFirstObject();
    while(pObj!=NULL)
    {
      if(pObj->GetTimeStamp()!=timeStamp)
      {
        ++at(lvl).second;
        pObj->SetTimeStamp(timeStamp);
      }
      pObj = pObj->GetNextQTreeNodeObject(pNode);
    }
  }
};

void Scene::ReportInfo()
{
  SceneInfo info;
  info.Get(&m_QTreeRoot, 0, GenerateTimeStamp());
  Log::Info(" lv |  World Space Size   | Nodes |  Obj. \n"
            "----+---------------------+-------+-------\n");
  Vec2 size = m_QTreeRoot.GetAABB().Size();
  for(size_t i=0; i<info.size(); ++i)
  {
    Log::Info(" %2d | %8.2f x %-8.2f | %5d | %5d \n", i, size.x, size.y, info[i].first, info[i].second);
    size *= 0.5f;
  }
}
