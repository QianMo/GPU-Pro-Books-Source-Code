#ifndef __TRANSFORM_GIZMO_H
#define __TRANSFORM_GIZMO_H

#include "DebugRenderer.h"
#include "Frustum.h"
#include "Scene/Camera.h"
#include "TextureLoader/Texture11.h"

template<class T>
class _TransformGizmo : public MathLibObject
{
public:
  _TransformGizmo() : m_Transform(Mat4x4::Identity()), m_TranslationAxis(-1), m_RotationAxis(-1)
  {
  }
  void Draw(DebugRenderer& drm)
  {
    Mat4x4 drmTransform = drm.GetTransform();
    Mat4x4 localGizmoTransform = Mat4x4::OBBSetScalingD3D(m_Transform, 1.0f)*drmTransform;

    static const Vec4 c_ColorX(1.0f, 0, 0, 0.5f);
    static const Vec4 c_ColorY(0, 1.0f, 0, 0.5f);
    static const Vec4 c_ColorZ(0, 0, 1.0f, 0.5f);
    static const Vec4 c_ColorSelected(1.0f);

    Vec4 axisColorX = m_TranslationAxis==0 ? c_ColorSelected : c_ColorX;
    Vec4 axisColorY = m_TranslationAxis==1 ? c_ColorSelected : c_ColorY;
    Vec4 axisColorZ = m_TranslationAxis==2 ? c_ColorSelected : c_ColorZ;
    drm.SetTransform(Mat4x4::Scaling(static_cast<T*>(this)->GetHandleLength())*localGizmoTransform);
    drm.SetContourColor(axisColorX); drm.PushLine<Vec3>(Vec3::Zero(), c_XAxis);
    drm.SetContourColor(axisColorY); drm.PushLine<Vec3>(Vec3::Zero(), c_YAxis);
    drm.SetContourColor(axisColorZ); drm.PushLine<Vec3>(Vec3::Zero(), c_ZAxis);
    drm.SetContourColor(Vec4::Zero());
    drm.SetFillColor(axisColorX); drm.SetTransform(Mat4x4::ScalingTranslationD3D(Vec3(static_cast<T*>(this)->GetArrowSize()) + (static_cast<T*>(this)->GetArrowLength() - static_cast<T*>(this)->GetArrowSize())*c_XAxis, static_cast<T*>(this)->GetHandleLength()*c_XAxis)*localGizmoTransform); drm.PushCone(c_XAxis, 8);
    drm.SetFillColor(axisColorY); 
    drm.SetTransform(Mat4x4::ScalingTranslationD3D(Vec3(static_cast<T*>(this)->GetArrowSize()) + (static_cast<T*>(this)->GetArrowLength() - static_cast<T*>(this)->GetArrowSize())*c_YAxis, static_cast<T*>(this)->GetHandleLength()*c_YAxis)*localGizmoTransform); 
    drm.PushCone(c_YAxis, 8);
    drm.SetFillColor(axisColorZ); drm.SetTransform(Mat4x4::ScalingTranslationD3D(Vec3(static_cast<T*>(this)->GetArrowSize()) + (static_cast<T*>(this)->GetArrowLength() - static_cast<T*>(this)->GetArrowSize())*c_ZAxis, static_cast<T*>(this)->GetHandleLength()*c_ZAxis)*localGizmoTransform); drm.PushCone(c_ZAxis, 8);

    Vec4 circleColorX = m_RotationAxis==0 ? c_ColorSelected : c_ColorX;
    Vec4 circleColorY = m_RotationAxis==1 ? c_ColorSelected : c_ColorY;
    Vec4 circleColorZ = m_RotationAxis==2 ? c_ColorSelected : c_ColorZ;
    drm.SetTransform(Mat4x4::Scaling(static_cast<T*>(this)->GetCircleRadius())*localGizmoTransform);
    drm.SetContourColor(circleColorX); drm.PushCircle(c_XAxis, 20);
    drm.SetContourColor(circleColorY); drm.PushCircle(c_YAxis, 20);
    drm.SetContourColor(circleColorZ); drm.PushCircle(c_ZAxis, 20);

    drm.SetTransform(drmTransform);
  }
  void SetTransform(const Mat4x4& tm)
  {
    m_Transform = tm;
    const Mat4x4 c_OBBScale[] =
    {
      Mat4x4::ScalingTranslationD3D(Vec3(0.5f*(static_cast<T*>(this)->GetHandleLength() + static_cast<T*>(this)->GetArrowLength()), static_cast<T*>(this)->GetArrowSize(), static_cast<T*>(this)->GetArrowSize()), Vec3(0.5f*(static_cast<T*>(this)->GetHandleLength() + static_cast<T*>(this)->GetArrowLength()), 0, 0)),
      Mat4x4::ScalingTranslationD3D(Vec3(static_cast<T*>(this)->GetArrowSize(), 0.5f*(static_cast<T*>(this)->GetHandleLength() + static_cast<T*>(this)->GetArrowLength()), static_cast<T*>(this)->GetArrowSize()), Vec3(0, 0.5f*(static_cast<T*>(this)->GetHandleLength() + static_cast<T*>(this)->GetArrowLength()), 0)),
      Mat4x4::ScalingTranslationD3D(Vec3(static_cast<T*>(this)->GetArrowSize(), static_cast<T*>(this)->GetArrowSize(), 0.5f*(static_cast<T*>(this)->GetHandleLength() + static_cast<T*>(this)->GetArrowLength())), Vec3(0, 0, 0.5f*(static_cast<T*>(this)->GetHandleLength() + static_cast<T*>(this)->GetArrowLength()))),
    };
    Mat4x4 rotTrans = Mat4x4::OBBSetScalingD3D(m_Transform, 1.0f);
    for(int i=0; i<3; ++i)
    {
      m_TranslationGizmoAxisOBB[i] = c_OBBScale[i]*rotTrans;
      m_TranslationGizmoAxisBRadius[i] = GetBSphereRadius(m_TranslationGizmoAxisOBB[i]);
    }
    Mat4x4 tmp = Mat4x4::Transpose(rotTrans);
    m_Planes = Mat4x4::SetTranslationD3D(tmp, -Vec3::Vector(m_Transform.r[3])*tmp);
  }

  const Mat4x4& GetTransform() const { return m_Transform; }
  const Vec3 GetPosition() const { return m_Transform.r[3]; }
  
protected:
  Mat4x4 m_Transform;
  Mat4x4 m_Planes;
  Mat4x4 m_TranslationGizmoAxisOBB[3];
  float m_TranslationGizmoAxisBRadius[3];
  int m_TranslationAxis;
  int m_RotationAxis;
};

class TransformGizmo : public _TransformGizmo<TransformGizmo>
{
public:
  static finline float GetHandleLength()    { return 1.0f; }
  static finline float GetArrowSize()       { return 0.05f; }
  static finline float GetArrowLength()     { return 0.2f; }
  static finline float GetCircleRadius()    { return 0.7f; }
  static finline float GetCircleThickness() { return 0.08f; }

  friend class TransformGizmoController;
};

class TransformGizmoController : public MathLibObject
{
public:
  TransformGizmoController() : m_TranslationGizmo(NULL), m_RotationGizmo(NULL), m_DragMode(false)
  {
  }
  bool ProcessWndMsg(UINT uMsg, WPARAM wParam, LPARAM lParam, Camera& camera)
  {
    if(uMsg==WM_MOUSEMOVE || uMsg==WM_LBUTTONDOWN || uMsg==WM_LBUTTONUP)
    {
      m_MouseCoord = Vec4i(LOWORD(lParam), HIWORD(lParam), 0, 0);

      Vec2 wh((float)Platform::GetBackBufferRT()->GetDesc().Width,  (float)Platform::GetBackBufferRT()->GetDesc().Height);
      Mat4x4 invProj = Mat4x4::ScalingTranslationD3D(Vec3(2.0f/wh.x, -2.0f/wh.y, 1), Vec3(-1.0f, 1.0f, 0))*camera.GetProjectionInverse();
      Vec2 mouseCoord = Vec4i::Convert(m_MouseCoord);
      Vec3 lb = Vec3::Project(Vec2(wh - mouseCoord - Vec2(0.5f)), invProj);
      Vec3 rt = Vec3::Project(Vec2(wh - mouseCoord + Vec2(0.5f)), invProj);
      Mat4x4 projMat = Mat4x4::ProjectionD3D(lb.x, rt.x, lb.y, rt.y, camera.GetNear(), camera.GetFar());
      m_RayFrustum = Frustum::FromViewProjectionMatrixD3D(camera.GetViewMatrix()*projMat);
      m_MouseRay.c = camera.GetPosition();
      m_MouseRay.d = Vec4::Normalize(0.5f*(lb + rt))*camera.GetViewMatrixInverse();
      m_MouseRayDir = Vec4::Normalize(Vec3::Project(mouseCoord, invProj))*camera.GetViewMatrixInverse();

      if(uMsg==WM_LBUTTONDOWN) { m_DragMode = true; m_DragOffset = FLT_MAX; }
      if(uMsg==WM_LBUTTONUP) m_DragMode = false;
      if(!m_DragMode) m_TranslationGizmo = m_RotationGizmo = NULL;
      return true;
    }
    return false;
  }
  template<class T>
  bool ProcessGizmo(T& g)
  {
    if(m_DragMode)
    {
      if(m_TranslationGizmo==&g)
      {
        Vec3 a = m_DragStartTransform.r[3];
        Vec3 b = Vec3::Normalize(g.m_Transform.r[g.m_TranslationAxis]);
        float BdotD = Vec3::Dot(b, m_MouseRay.d);
        float t = (Vec3::Dot(b, a - m_MouseRay.c) + Vec3::Dot(m_MouseRay.d, m_MouseRay.c - a)*BdotD)/(1.0f - BdotD*BdotD);
        if(m_DragOffset==FLT_MAX)
          m_DragOffset = t;
        g.SetTransform(Mat4x4::SetTranslationD3D(g.GetTransform(), a + (t - m_DragOffset)*b));
        return true;
      }
      else if(m_RotationGizmo==&g)
      {
        Vec4 plane = Mat4x4::Transpose(g.m_Planes).r[g.m_RotationAxis];
        float f = -Vec4::Dot(Vec3::Point(m_MouseRay.c), plane)/Vec3::Dot(m_MouseRayDir, plane);
        Vec3 d = m_MouseRay.c - g.GetPosition() + f*m_MouseRayDir;
        Vec4 axis = Vec4::Zero();
        axis[g.m_RotationAxis] = 1.0f;
        Vec3 n = axis*m_DragStartTransform;
        Vec3 t = GetArbitraryOrthogonalVector(n);
        Vec3 b = Vec3::Cross(t, n);
        Vec3 projD = d*Mat4x4::Transpose(Mat4x4(t, b, n, c_WAxis));
        float a = atan2f(projD.y, projD.x);
        if(m_DragOffset==FLT_MAX)
          m_DragOffset = a;
        Mat4x4 tm = Mat4x4::SetTranslationD3D(m_DragStartTransform, Vec3::Zero())*Quat::AsMatrixD3D(Quat(n, m_DragOffset - a));
        g.SetTransform(Mat4x4::SetTranslationD3D(tm, g.GetPosition()));
        return true;
      }
    }
    else
    {
      g.m_TranslationAxis = g.m_RotationAxis = -1;
      for(int i=0; i<3; ++i)
        if(m_RayFrustum.IsIntersecting(g.m_TranslationGizmoAxisOBB[i], g.m_TranslationGizmoAxisBRadius[i]))
          { g.m_TranslationAxis = i; m_TranslationGizmo = &g; m_DragStartTransform = g.GetTransform(); break; }
      if(g.m_TranslationAxis<0)
      {
        Vec4 t = -(m_MouseRay.c*g.m_Planes)*Vec4::ApproxRcp(Vec3::Vector(m_MouseRayDir)*g.m_Planes);
        Vec4 d = m_MouseRay.c - g.GetPosition();
        Mat4x4 m = Mat4x4::Transpose(Mat4x4(d + Vec4::Swizzle<x,x,x,x>(t)*m_MouseRayDir,
                                            d + Vec4::Swizzle<y,y,y,y>(t)*m_MouseRayDir,
                                            d + Vec4::Swizzle<z,z,z,z>(t)*m_MouseRayDir,
                                            Vec4::Zero()));
        Vec4 lenSq = m.r[0]*m.r[0] + m.r[1]*m.r[1] + m.r[2]*m.r[2];
        const Vec4 c_Low((g.GetCircleRadius() - g.GetCircleThickness())*(g.GetCircleRadius() - g.GetCircleThickness()));
        const Vec4 c_Hi ((g.GetCircleRadius() + g.GetCircleThickness())*(g.GetCircleRadius() + g.GetCircleThickness()));
        unsigned mask = Vec4::AsMask(Vec4::CmpGreater(lenSq, c_Low) & Vec4::CmpLess(lenSq, c_Hi) & Vec4::CmpGreater(t, Vec4::Zero()));
        for(int i=0; i<3; ++i)
          if(mask & (1<<i))
            { g.m_RotationAxis = i; m_RotationGizmo = &g; m_DragStartTransform = g.GetTransform(); break; }
      }
    }
    return false;
  }

protected:
  Frustum m_RayFrustum;
  Vec4i m_MouseCoord;
  struct { Vec3 c, d; } m_MouseRay;
  Mat4x4 m_DragStartTransform;
  Vec3 m_MouseRayDir;
  void* m_TranslationGizmo;
  void* m_RotationGizmo;
  float m_DragOffset;
  bool m_DragMode;
};

#endif //#ifndef __TRANSFORM_GIZMO_H
