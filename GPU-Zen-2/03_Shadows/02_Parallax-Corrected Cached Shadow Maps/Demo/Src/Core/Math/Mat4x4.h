/* The functions <something>D3D (e.g. OrthoD3D) construct matrices for left 
   multiplication of the matrix with a vector, b = a*M.

   The functions OBB<something> (e.g. OBBSetScalingD3D) work with specific affine 
   transforms, which represent scaling followed by rotation followed by translation.
*/

#ifndef __MAT4X4_COMMON
#define __MAT4X4_COMMON

const Mat4x4 Mat4x4::Identity()
{
  return Mat4x4(c_XAxis, c_YAxis, c_ZAxis, c_WAxis);
}

finline const Mat4x4 operator + (const Mat4x4& a, const Mat4x4& b)
{
  return Mat4x4(a.r[0] + b.r[0], a.r[1] + b.r[1], a.r[2] + b.r[2], a.r[3] + b.r[3]);
}

finline const Mat4x4 operator - (const Mat4x4& a, const Mat4x4& b)
{
  return Mat4x4(a.r[0] - b.r[0], a.r[1] - b.r[1], a.r[2] - b.r[2], a.r[3] - b.r[3]);
}

finline float Mat4x4::FrobeniusNorm(const Mat4x4& m)
{
  return Vec4::Length(m.r[0]*m.r[0] + m.r[1]*m.r[1] + m.r[2]*m.r[2] + m.r[3]*m.r[3]);
}

const Mat4x4 Mat4x4::LookAtD3D(const Vec3& eye, const Vec3& target, const Vec3& up)
{
  Vec3 vz = Vec3::Normalize(target - eye);
  Vec3 vx = Vec3::Normalize(Vec3::Cross(up, vz));
  Vec3 vy = Vec3::Cross(vz, vx);
  Mat4x4 m = Mat4x4::Transpose(Mat4x4(Vec3::Vector(vx), Vec3::Vector(vy), Vec3::Vector(vz), Vec4::Zero()));
  return SetTranslationD3D(m, -(eye*m));
}

const Mat4x4 Mat4x4::OrthoD3D(float l, float r, float b, float t, float zn, float zf)
{
  Vec3 s = Vec3::Rcp(Vec3(l - r, b - t, zn - zf));
  return ScalingTranslationD3D(s*Vec3::Constant<-2,-2,-1>(), s*Vec3(l + r, t + b, zn));
}

const Mat4x4 Mat4x4::ProjectionD3D(float l, float r, float b, float t, float zn, float zf)
{
  Vec3 s = Vec3::Rcp(Vec3(r - l, t - b, zn - zf));
  return Mat4x4::Transpose(Mat4x4(Vec4::Swizzle<x,x,x,x>(s)*Vec4(2.0f*zn, 0, l + r, 0),
                                  Vec4::Swizzle<y,y,y,y>(s)*Vec4(0, 2.0f*zn, b + t, 0),
                                  Vec4::Swizzle<z,z,z,z>(s)*Vec4(0,  0, -zf, zn*zf),
                                  c_ZAxis));
}

const Mat4x4 Mat4x4::ProjectionD3D(float fovY, float aspect, float zn, float zf)
{
  Vec4 sinFov, cosFov;
  Vec4::SinCos(Vec4(fovY*0.5f), sinFov, cosFov);
  Vec3 s = Vec3::Rcp(Vec3(sinFov.x*aspect, sinFov.x, zn - zf));
  return Mat4x4::Transpose(Mat4x4(Vec4::Swizzle<x,x,x,x>(s)*Vec4(cosFov.x, 0, 0, 0),
                                  Vec4::Swizzle<y,y,y,y>(s)*Vec4(0, cosFov.x, 0, 0),
                                  Vec4::Swizzle<z,z,z,z>(s)*Vec4(0, 0, -zf, zn*zf),
                                  c_ZAxis));
}

const Mat4x4 Mat4x4::ProjectionGL(float fovY, float aspect, float zn, float zf)
{
  Vec4 sinFov, cosFov;
  Vec4::SinCos(Vec4(fovY*0.5f), sinFov, cosFov);
  Vec3 s = Vec3::Rcp(Vec3(sinFov.x*aspect, sinFov.x, zf - zn));
  return Mat4x4::Transpose(Mat4x4(Vec4::Swizzle<x,x,x,x>(s)*Vec4(cosFov.x, 0, 0, 0),
                                  Vec4::Swizzle<y,y,y,y>(s)*Vec4(0, cosFov.x, 0, 0),
                                  Vec4::Swizzle<z,z,z,z>(s)*Vec4(0, 0, (zf+zn), -2*zn*zf),
                                  c_ZAxis));
}

const Mat4x4 Mat4x4::SetTranslationD3D(const Mat4x4& a, const Vec3& b)
{
  return Mat4x4(a.r[0], a.r[1], a.r[2], Vec3::Point(b));
}

const Quat Mat4x4::AsQuaternion(const Mat4x4& obb)
{
  return AsQuaternion(OBBSetScalingD3D(obb, Vec3(1.0f)), 1.0f);
}

const Mat4x4 Mat4x4::Scaling(const Vec3& a)
{
  return Mat4x4(Vec4::Swizzle<x,w,w,w>(a), Vec4::Swizzle<w,y,w,w>(a), Vec4::Swizzle<w,w,z,w>(a), c_WAxis);
}

const Mat4x4 Mat4x4::TranslationD3D(const Vec3& a)
{
  return Mat4x4(c_XAxis, c_YAxis, c_ZAxis, Vec3::Point(a));
}

const Mat4x4 Mat4x4::ScalingTranslationD3D(const Vec3& a, const Vec3& b)
{
  return Mat4x4(Vec4::Swizzle<x,w,w,w>(a), Vec4::Swizzle<w,y,w,w>(a), Vec4::Swizzle<w,w,z,w>(a), Vec3::Point(b));
}

void Mat4x4::OBBtoAABB_D3D(const Mat4x4& obb, Vec3& AABBMin, Vec3& AABBMax)
{
  Vec4 r = Vec4::Abs(obb.r[0]) + Vec4::Abs(obb.r[1]) + Vec4::Abs(obb.r[2]);
  AABBMin = obb.r[3] - r;
  AABBMax = obb.r[3] + r;
}

const Mat4x4 Mat4x4::OBBSetScalingD3D(const Mat4x4& obb, const Vec3& s)
{
  Mat4x4 t0 = Mat4x4::Transpose(obb);
  Vec4 t1 = s*Vec4::Rsqrt(t0.r[0]*t0.r[0] + t0.r[1]*t0.r[1] + t0.r[2]*t0.r[2]);
  return Mat4x4(Vec4::Swizzle<x,x,x,x>(t1)*obb.r[0],
                Vec4::Swizzle<y,y,y,y>(t1)*obb.r[1],
                Vec4::Swizzle<z,z,z,z>(t1)*obb.r[2],
                obb.r[3]);
}

const Mat4x4 Mat4x4::OBBInverseD3D(const Mat4x4& obb)
{
  Mat4x4 t0 = Mat4x4::Transpose(obb);
  Vec4 t1 = Vec4::Rcp(t0.r[0]*t0.r[0] + t0.r[1]*t0.r[1] + t0.r[2]*t0.r[2]);
  Mat4x4 t2 = Mat4x4::Transpose(Mat4x4(Vec4::Swizzle<x,x,x,x>(t1)*obb.r[0],
                                       Vec4::Swizzle<y,y,y,y>(t1)*obb.r[1],
                                       Vec4::Swizzle<z,z,z,z>(t1)*obb.r[2],
                                       c_WAxis));
  return SetTranslationD3D(t2, -Vec3(obb.r[3])*t2);
}

#endif //#ifndef __MAT4X4_COMMON
