#ifndef __QTREE
#define __QTREE

#include <vector>
#include "../Math/Math.h"

template<class T> class QTreeNode : public MathLibObject
{
public:
  QTreeNode(const AABB2D& BBox) : m_Parent(NULL), m_BBox(BBox), m_Depth(0), m_ChildIndex(0)
  {
    memset(m_Children, 0, sizeof(m_Children));
  }
  QTreeNode(T* pParent, unsigned short childIndex) : m_Parent(pParent), m_ChildIndex(childIndex)
  {
    _ASSERT(m_Parent->m_Children[childIndex]==NULL && "parent node has a child");
    m_Parent->m_Children[m_ChildIndex] = static_cast<T*>(this);
    m_BBox = m_Parent->GetChildBBox(childIndex);
    m_Depth = m_Parent->m_Depth + 1;
    memset(m_Children, 0, sizeof(m_Children));
  }
  ~QTreeNode()
  {
    for(int i=0; i<4; ++i)
      delete m_Children[i];
    if(m_Parent!=NULL)
      m_Parent->m_Children[m_ChildIndex] = NULL;
  }
  finline AABB2D GetChildBBox(unsigned i) const
  {
    static const Vec2 s_Offset[] = { Vec2(0,0), Vec2(1,0), Vec2(0,1), Vec2(1,1) };
    Vec2 hsize = 0.5f*m_BBox.Size();
    Vec2 min = m_BBox.GetMin() + s_Offset[i]*hsize;
    return AABB2D(min, min + hsize);
  }
  void RemoveEmptyChildren()
  {
    for(int i=0; i<4; ++i)
      if(m_Children[i]!=NULL)
        RemoveEmptyNodes(m_Children[i]);
  }
  void DestroyIfEmpty()
  {
    RemoveEmptyNodes(this);
  }

  finline const AABB2D& GetAABB() const { return m_BBox; }
  finline T* GetParent() const { return m_Parent; }
  finline T* GetChild(unsigned i) const { return m_Children[i]; }
  finline unsigned GetChildIndex() const { return m_ChildIndex; }
  finline unsigned GetDepth() const { return m_Depth; }

protected:
  AABB2D m_BBox;
  T* m_Parent;
  T* m_Children[4];
  unsigned short m_Depth;
  unsigned short m_ChildIndex;

  static void RemoveEmptyNodes(T* pNode)
  {
    bool hasChildren = false;
    for(int i=0; i<4; ++i)
    {
      if(pNode->m_Children[i]!=NULL)
        RemoveEmptyNodes(pNode->m_Children[i]);
      hasChildren |= (pNode->m_Children[i]!=NULL);
    }
    if(!hasChildren && pNode->GetFirstObject()==NULL)
      delete pNode;
  }
};

template<class T, class NODE> class QTreeNodeObject
{
public:
  QTreeNodeObject()
  {
    memset(m_BelongsTo, 0, sizeof(m_BelongsTo));
    memset(m_Next, 0, sizeof(m_Next));
  }
  ~QTreeNodeObject()
  {
    Remove();
  }
  void Assign(NODE* pNode, const AABB2D& objectAABB)
  {
    PushDown(pNode, objectAABB);
    for(int i=0; i<4; ++i)
    {
      if(m_BelongsTo[i]!=NULL)
      {
        m_Next[i] = m_BelongsTo[i]->m_FirstObject;
        m_BelongsTo[i]->m_FirstObject = static_cast<T*>(this);
      }
    }
  }
  void Remove()
  {
    for(int i=0; i<4; ++i)
    {
      if(m_BelongsTo[i]!=NULL)
      {
        if(m_BelongsTo[i]->m_FirstObject!=this)
        {
          QTreeNodeObject* p;
          for(p=m_BelongsTo[i]->m_FirstObject; p!=NULL; p=p->m_Next[i])
            if(p->m_Next[i]==this) { p->m_Next[i] = m_Next[i]; break; }
          _ASSERT(p!=NULL);
        }
        else
          m_BelongsTo[i]->m_FirstObject = m_Next[i];
      }
    }
    memset(m_BelongsTo, 0, sizeof(m_BelongsTo));
    memset(m_Next, 0, sizeof(m_Next));
  }
  finline T* GetNextQTreeNodeObject(NODE* pNode) const
  {
    for(int i=0; i<4; ++i)
      if(m_BelongsTo[i]==pNode)
        return m_Next[i];
    _ASSERT(!"this object does not belong to given node");
    return NULL;
  }

protected:
  NODE* m_BelongsTo[4];
  T* m_Next[4];

  void PushDown(NODE* pNode, const AABB2D& objectAABB)
  {
    const AABB2D& nodeAABB = pNode->GetAABB();
    float area = AABB2D::GetOverlapArea(nodeAABB, objectAABB);
    if(area>0)
    {
      if(!(objectAABB.Size() <= 0.5f*nodeAABB.Size()))
      {
        unsigned childIndex = pNode->GetChildIndex();
        if(m_BelongsTo[childIndex]==NULL ||
          AABB2D::GetOverlapArea(m_BelongsTo[childIndex]->GetAABB(), objectAABB)<area)
        {
          _ASSERT(m_BelongsTo[childIndex]==NULL);
          m_BelongsTo[childIndex] = pNode;
        }
      }
      else
      {
        for(unsigned short i=0; i<4; ++i)
        {
          if(pNode->GetChild(i)==NULL && AABB2D::IsIntersecting(objectAABB, pNode->GetChildBBox(i)))
            new NODE(pNode, i);
          if(pNode->GetChild(i)!=NULL)
            PushDown(pNode->GetChild(i), objectAABB);
        }
      }
    }
  }
};

#endif //#ifndef __QTREE
