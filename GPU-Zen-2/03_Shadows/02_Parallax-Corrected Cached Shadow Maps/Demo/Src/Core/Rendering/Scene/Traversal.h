#ifndef __SCENE_TRAVERSAL_H
#define __SCENE_TRAVERSAL_H

#include "SceneObject.h"
#include "../../Util/Frustum.h"

class SceneQTreeTraversal
{
public:
  SceneQTreeTraversal() : m_C0(1e+6f)
  {
  }
  void ProcessVisiblePrepare(const Mat4x4& viewProj, const Vec3& viewPos, unsigned timeStamp)
  {
    m_Frustum = Frustum::FromViewProjectionMatrixD3D(viewProj);
    m_ViewPos = viewPos;
    m_TimeStamp = timeStamp;
  }
  void SetContributionCullingScreenAreaThreshold(const Mat4x4& perspProj, float thresholdProjection)
  {
    if(thresholdProjection==0.0f)
    {
        m_C0 = 1e+6f;
    }
    else
    {
        _ASSERT(Mat4x4::Transpose(perspProj).r[3]==c_ZAxis && "perspective projection matrix is required");
        m_C0 = perspProj.e11*perspProj.e22*4.0f/(c_PI*thresholdProjection);
    }
  }
  template<class T, void (T::*ReportVisible)(SceneObject*)> void ProcessVisible(T* pCaller, SceneQTreeNode* pNode)
  {
    const AABB2D& nodeAABB = pNode->GetAABB();
    Vec3 min(nodeAABB.x, pNode->GetMinHeight(), nodeAABB.y);
    Vec3 max(nodeAABB.z, pNode->GetMaxHeight(), nodeAABB.w);
    Vec3 hsize = 0.5f*(max - min);
    Vec3 center = 0.5f*(max + min);
    Mat4x4 aabb = Mat4x4::ScalingTranslationD3D(hsize, center);
    if(m_Frustum.IsIntersecting(aabb))
    {
      if(Vec3::LengthSq(center - m_ViewPos) < m_C0*Vec3::LengthSq(hsize))
      {
        for(SceneObject* pObj=pNode->GetFirstObject(); pObj!=NULL; pObj=pObj->GetNextQTreeNodeObject(pNode))
          if(pObj->GetTimeStamp()!=m_TimeStamp && m_Frustum.IsIntersecting(pObj->GetOBB(), pObj->GetBSphereRadius()))
            if(Vec3::LengthSq(Vec3(pObj->GetOBB().r[3]) - m_ViewPos) < (m_C0*pObj->GetBSphereRadius()*pObj->GetBSphereRadius()))
              { pObj->SetTimeStamp(m_TimeStamp); (pCaller->*ReportVisible)(pObj); }
        for(int i=0; i<4; ++i)
          if(pNode->GetChild(i)!=NULL)
            ProcessVisible<T, ReportVisible>(pCaller, pNode->GetChild(i));
      }
    }
  }

private:
  Frustum m_Frustum;
  Vec3 m_ViewPos;
  unsigned m_TimeStamp;
  float m_C0;
};

#endif //#ifndef __SCENE_TRAVERSAL_H
