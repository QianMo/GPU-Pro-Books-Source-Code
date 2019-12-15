#ifndef __SCENE_OBJECT
#define __SCENE_OBJECT

#include "../../Util/QTree.h"
#include "../../Util/MemoryPool.h"

class SceneObject;
class Renderable;

class SceneQTreeNode : public QTreeNode<SceneQTreeNode>
{
  DECLARE_MEMORY_POOL();
public:
  SceneQTreeNode(const AABB2D& BBox) : 
    QTreeNode(BBox), m_FirstObject(NULL), m_MinHeight(FLT_MAX), m_MaxHeight(-FLT_MAX) { }
  SceneQTreeNode(SceneQTreeNode* pParent, unsigned short childIndex) : 
    QTreeNode(pParent, childIndex), m_FirstObject(NULL), m_MinHeight(FLT_MAX), m_MaxHeight(-FLT_MAX) { }

  finline SceneObject* GetFirstObject() const { return m_FirstObject; }
  finline float GetMinHeight() const { return m_MinHeight; }
  finline float GetMaxHeight() const { return m_MaxHeight; }
  finline float GetBCircleRadius() const { return m_BCircleRadius; }

protected:
  SceneObject* m_FirstObject;
  float m_MinHeight;
  float m_MaxHeight;
  float m_BCircleRadius;

  friend class SceneObject;
  friend class QTreeNodeObject<SceneObject, SceneQTreeNode>;
};

class SceneObject : public QTreeNodeObject<SceneObject, SceneQTreeNode>, public MathLibObject
{
public:
  SceneObject(const Mat4x4& aabb = Mat4x4::Identity(), SceneObject* pParent = NULL, SceneQTreeNode* pQTreeRoot = NULL);
  virtual ~SceneObject();

  virtual Renderable* PrepareToRender() { return NULL; }

  virtual void OnTransformChanged();
  virtual void UpdateBSphereRadius();

  void SetTransform(const Mat4x4&);
  void SetLocalSpaceTransform(const Mat4x4& tm) { SetTransform(m_Parent->m_Transform*tm); }
  void Assign(SceneQTreeNode*, const AABB2D&);

  finline SceneObject* GetParent() const { return m_Parent; }
  finline SceneObject* GetFirstChild() const { return m_FirstChild; }
  finline SceneObject* NextSibling() const { return m_NextSibling; }
  finline const Mat4x4 GetTransform() const { return m_Transform; }
  finline const Mat4x4 GetOBB() const { return m_OBB; }
  finline float GetBSphereRadius() const { return m_BSphereRadius; }
  finline SceneQTreeNode* GetQTreeRoot() const { return m_QTreeRoot; }
  finline void SetPosition(const Vec3& pos) { SetTransform(Mat4x4::SetTranslationD3D(m_Transform, pos)); }
  finline const Vec3 GetPosition() const { return m_Transform.r[3]; }
  finline unsigned GetTimeStamp() const { return m_TimeStamp; }
  finline void SetTimeStamp(unsigned t) { m_TimeStamp = t; }

protected:
  Mat4x4 m_Transform;
  Mat4x4 m_OBB;
  SceneQTreeNode* m_QTreeRoot;
  SceneObject* m_Parent;
  SceneObject* m_FirstChild;
  SceneObject* m_NextSibling;
  float m_BSphereRadius;
  unsigned m_TimeStamp;
};

#endif //#ifndef __SCENE_OBJECT
