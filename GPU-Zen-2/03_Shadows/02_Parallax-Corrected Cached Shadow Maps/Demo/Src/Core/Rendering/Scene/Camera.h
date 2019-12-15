#ifndef __CAMERA_H
#define __CAMERA_H

#include "../../Math/Math.h"

class Camera : public MathLibObject
{
public:
  Camera() : m_ProjMat(Mat4x4::Identity()), m_InvProj(Mat4x4::Identity())
  {
    SetViewMatrix(Mat4x4::Identity());
    OnUpdate();
  }
  void SetViewMatrix(const Mat4x4& viewMat)
  {
    m_ViewMat = viewMat;
    m_InvView = Mat4x4::Inverse(m_ViewMat);
    Mat4x4 m = Mat4x4::OBBSetScalingD3D(m_InvView, Vec3(1.0f)); // works only for uniform view matrix scaling
    m_RightVec = m.r[0];
    m_UpVec = m.r[1];
    m_FrontVec = m.r[2];
    OnUpdate();
  }
  void SetProjection(const Mat4x4& projMat)
  {
    m_ProjMat = projMat;
    m_Near = -m_ProjMat.e43/m_ProjMat.e33;
    m_Far = projMat.e44==0 ? m_ProjMat.e43/(1.0f - m_ProjMat.e33) : (1.0f - m_ProjMat.e43)/m_ProjMat.e33;
    m_InvProj = Mat4x4::Inverse(m_ProjMat);
    OnUpdate();
  }

  finline const Mat4x4& GetViewMatrix() const { return m_ViewMat; }
  finline const Mat4x4& GetProjection() const { return m_ProjMat; }
  finline const Mat4x4& GetViewProjection() const { return m_ViewProj; }
  finline const Mat4x4& GetViewMatrixInverse() const { return m_InvView; }
  finline const Mat4x4& GetProjectionInverse() const { return m_InvProj; }
  finline const Mat4x4& GetViewProjectionInverse() const { return m_InvViewProj; }
  finline const Vec3& GetFrontVector() const { return m_FrontVec; }
  finline const Vec3& GetUpVector() const { return m_UpVec; }
  finline const Vec3& GetRightVector() const { return m_RightVec; }
  finline const Vec3 GetPosition() const { return m_InvView.r[3]; }
  finline const Vec4& GetFrustumPlane(unsigned i) const { return m_FrustumPlanes[i]; }
  finline float GetNear() const { return m_Near; }
  finline float GetFar() const { return m_Far; }

protected:
  Mat4x4 m_ViewMat, m_ProjMat, m_ViewProj;
  Mat4x4 m_InvView, m_InvProj, m_InvViewProj;
  Vec3 m_FrontVec, m_UpVec, m_RightVec;
  Vec4 m_FrustumPlanes[6];
  float m_Near, m_Far;

  static finline const Vec4 PlaneNormalize(const Vec4& v)
  {
    return v/Vec3::Length(Vec3(v));
  }
  finline void OnUpdate()
  {
    m_ViewProj = m_ViewMat*m_ProjMat;
    m_InvViewProj = m_InvProj*m_InvView;

    m_FrustumPlanes[0] = PlaneNormalize(Vec4(m_ViewProj.e14 + m_ViewProj.e11, m_ViewProj.e24 + m_ViewProj.e21, m_ViewProj.e34 + m_ViewProj.e31, m_ViewProj.e44 + m_ViewProj.e41));
    m_FrustumPlanes[1] = PlaneNormalize(Vec4(m_ViewProj.e14 - m_ViewProj.e11, m_ViewProj.e24 - m_ViewProj.e21, m_ViewProj.e34 - m_ViewProj.e31, m_ViewProj.e44 - m_ViewProj.e41));
    m_FrustumPlanes[2] = PlaneNormalize(Vec4(m_ViewProj.e14 - m_ViewProj.e12, m_ViewProj.e24 - m_ViewProj.e22, m_ViewProj.e34 - m_ViewProj.e32, m_ViewProj.e44 - m_ViewProj.e42));
    m_FrustumPlanes[3] = PlaneNormalize(Vec4(m_ViewProj.e14 + m_ViewProj.e12, m_ViewProj.e24 + m_ViewProj.e22, m_ViewProj.e34 + m_ViewProj.e32, m_ViewProj.e44 + m_ViewProj.e42));
    m_FrustumPlanes[4] = PlaneNormalize(Vec4(m_ViewProj.e13, m_ViewProj.e23, m_ViewProj.e33, m_ViewProj.e43));
    m_FrustumPlanes[5] = PlaneNormalize(Vec4(m_ViewProj.e14 - m_ViewProj.e13, m_ViewProj.e24 - m_ViewProj.e23, m_ViewProj.e34 - m_ViewProj.e33, m_ViewProj.e44 - m_ViewProj.e43));
  }
};

#endif //#ifndef __CAMERA_H
