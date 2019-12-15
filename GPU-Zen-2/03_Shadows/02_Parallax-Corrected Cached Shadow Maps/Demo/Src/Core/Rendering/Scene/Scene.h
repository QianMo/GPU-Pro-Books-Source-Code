#ifndef __SCENE
#define __SCENE

#include "SceneObject.h"
#include "TextureLoader/TextureLoader.h"
#include "Mesh/Mesh.h"

namespace rapidxml
{
  template<class> class xml_node;
}

class MeshFileDesc
{
public:
  MeshFileDesc(const char* pszFileName) : m_FileName(pszFileName) { }
  const std::string& GetFileName() const { return m_FileName; }

  typedef MeshFileDesc Description;
  typedef Mesh* CacheEntry;
  typedef TextureLoader* UserParam;

  static finline bool Allocate(const UserParam& pTexLdr, const Description& d, CacheEntry& e)
  {
    e = new Mesh();
    if(e!=NULL && e->LoadXML(d.m_FileName.c_str(), *pTexLdr))
      return true;
    Free(pTexLdr, e);
    return false;
  }
  static finline void Free(const UserParam&, CacheEntry& e)
  {
    delete e;
    e = NULL;
  }

protected:
  std::string m_FileName;
};

class Scene : public MathLibObject
{
public:
  Scene(const Vec4& AABBMin = Vec4::Zero(), const Vec4& AABBMax = Vec4::Zero());
  ~Scene();
  bool LoadXML(const char*);
  void ReportInfo();

  finline SceneQTreeNode* GetQTreeRoot() { return &m_QTreeRoot; }
  finline SceneObject* GetSceneRoot() { return &m_RootObject; }
  finline unsigned GenerateTimeStamp() { return ++m_TimeStamp; }

protected:
  typedef Cache<MeshFileDesc, FileResourceDescCompare<MeshFileDesc> > Meshes;

  Meshes m_Meshes;
  SceneQTreeNode m_QTreeRoot;
  SceneObject m_RootObject;
  unsigned m_TimeStamp;

  bool ParseNode(SceneObject*, rapidxml::xml_node<char>*);
};

#endif //#ifndef __SCENE
