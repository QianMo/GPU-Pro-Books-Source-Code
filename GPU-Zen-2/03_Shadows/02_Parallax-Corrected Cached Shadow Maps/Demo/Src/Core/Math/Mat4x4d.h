#ifndef __MAT4X4d_COMMON
#define __MAT4X4d_COMMON

finline const Mat4x4d operator + (const Mat4x4d& a, const Mat4x4d& b)
{
  return Mat4x4d(a.r[0] + b.r[0], a.r[1] + b.r[1], a.r[2] + b.r[2], a.r[3] + b.r[3]);
}

finline const Mat4x4d operator - (const Mat4x4d& a, const Mat4x4d& b)
{
  return Mat4x4d(a.r[0] - b.r[0], a.r[1] - b.r[1], a.r[2] - b.r[2], a.r[3] - b.r[3]);
}

finline const Mat4x4d operator * (const Mat4x4d& a, const Mat4x4d& b)
{
  return Mat4x4d(a.r[0].x*b.r[0] + a.r[0].y*b.r[1] + a.r[0].z*b.r[2] + a.r[0].w*b.r[3],
                 a.r[1].x*b.r[0] + a.r[1].y*b.r[1] + a.r[1].z*b.r[2] + a.r[1].w*b.r[3],
                 a.r[2].x*b.r[0] + a.r[2].y*b.r[1] + a.r[2].z*b.r[2] + a.r[2].w*b.r[3],
                 a.r[3].x*b.r[0] + a.r[3].y*b.r[1] + a.r[3].z*b.r[2] + a.r[3].w*b.r[3]);
}

finline const Vec4d operator * (const Vec4d& v, const Mat4x4d& m)
{
  return v.x*m.r[0] + v.y*m.r[1] + v.z*m.r[2] + v.w*m.r[3];
}

finline const Vec3d operator * (const Vec3d& v, const Mat4x4d& m)
{
  return v.x*m.r[0] + v.y*m.r[1] + v.z*m.r[2] + m.r[3];
}

finline const Vec2d operator * (const Vec2d& v, const Mat4x4d& m)
{
  return v.x*m.r[0] + v.y*m.r[1] + m.r[3];
}

const Mat4x4d Mat4x4d::SetTranslationD3D(const Mat4x4d& a, const Vec3d& b)
{
  return Mat4x4d(a.r[0], a.r[1], a.r[2], Vec3d::Point(b));
}

const Mat4x4d Mat4x4d::LookAtD3D(const Vec3d& eye, const Vec3d& target, const Vec3d& up)
{
  Vec3d vz = Vec3d::Normalize(target - eye);
  Vec3d vx = Vec3d::Normalize(Vec3d::Cross(up, vz));
  Vec3d vy = Vec3d::Cross(vz, vx);
  Mat4x4d m = Mat4x4d::Transpose(Mat4x4d(Vec3d::Vector(vx), Vec3d::Vector(vy), Vec3d::Vector(vz), Vec4d::Zero()));
  return SetTranslationD3D(m, -(eye*m));
}

const Mat4x4d Mat4x4d::TranslationD3D(const Vec3d& a)
{
  return SetTranslationD3D(Mat4x4d::Identity(), a);
}

const Mat4x4d Mat4x4d::OrthoD3D(double l, double r, double b, double t, double zn, double zf)
{
  Vec3d lbn(l, b, zn);
  Vec3d rtf(r, t, zf);
  Vec3d s = lbn - rtf;
  return ScalingTranslationD3D(Vec3d::Constant<-2,-2,-1>()/s, (lbn + Vec2d(rtf))/s);
}

#endif //#ifndef __MAT4X4d_COMMON
