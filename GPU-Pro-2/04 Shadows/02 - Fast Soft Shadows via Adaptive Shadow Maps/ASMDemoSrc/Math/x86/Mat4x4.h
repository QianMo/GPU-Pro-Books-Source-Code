/* MSVC SSE math library implementation.

   Vector-matrix multiplicatons here are point transforms:
     vec2*mat4x4 == Vec2::Point(vec2)*mat4x4
     vec3*mat4x4 == Vec3::Point(vec3)*mat4x4

   Pavlo Turchyn <pavlo@nichego-novogo.net> Aug 2010 */

#ifndef __MAT4X4
#define __MAT4X4

class Mat4x4 : public MathLibObject
{
public:
   union
   {
     struct { float e11, e12, e13, e14, e21, e22, e23, e24, e31, e32, e33, e34, e41, e42, e43, e44; };
     __m128 r[4];
   };

   finline Mat4x4() { }
   finline Mat4x4(const __m128& a, const __m128& b, const __m128& c, const __m128& d) { r[0] = a; r[1] = b; r[2] = c; r[3] = d; }
   finline Mat4x4(const Mat4x4& a) { r[0] = a.r[0]; r[1] = a.r[1]; r[2] = a.r[2]; r[3] = a.r[3]; }
   finline Mat4x4(const float* a) { r[0] = _mm_loadu_ps(a); r[1] = _mm_loadu_ps(&a[4]); r[2] = _mm_loadu_ps(&a[8]); r[3] = _mm_loadu_ps(&a[12]); }

   finline const Mat4x4& operator = (const Mat4x4& a) { r[0] = a.r[0]; r[1] = a.r[1]; r[2] = a.r[2]; r[3] = a.r[3]; return *this; }

   static finline const Mat4x4 Identity()
   {
     return Mat4x4(SWZ_WXXX(c_WOne), SWZ_XWXX(c_WOne), SWZ_XXWX(c_WOne), c_WOne);
   }
   static finline const Mat4x4 Scaling(const Vec3& a)
   {
     return Mat4x4(SWZ_XWWW(a.r), SWZ_WYWW(a.r), SWZ_WWZW(a.r), c_WOne);
   }
   static finline const Mat4x4 TranslationD3D(const Vec3& a)
   {
     return Mat4x4(SWZ_WXXX(c_WOne), SWZ_XWXX(c_WOne), SWZ_XXWX(c_WOne), _mm_or_ps(a.r, c_WOne));
   }
   static finline const Mat4x4 ScalingTranslationD3D(const Vec3& a, const Vec3& b)
   {
     return Mat4x4(SWZ_XWWW(a.r), SWZ_WYWW(a.r), SWZ_WWZW(a.r), _mm_or_ps(b.r, c_WOne));
   }
   static finline const Mat4x4 SetTranslationD3D(const Mat4x4& a, const Vec3& b)
   {
     return Mat4x4(a.r[0], a.r[1], a.r[2], _mm_or_ps(b.r, c_WOne));
   }
   static finline const Mat4x4 Transpose(const Mat4x4& a)
   {
     __m128 t0 = _mm_unpacklo_ps(a.r[0], a.r[1]);
     __m128 t1 = _mm_unpackhi_ps(a.r[0], a.r[1]);
     __m128 t2 = _mm_unpacklo_ps(a.r[2], a.r[3]);
     __m128 t3 = _mm_unpackhi_ps(a.r[2], a.r[3]);
     return Mat4x4(_mm_movelh_ps(t0, t2),
                   _mm_movehl_ps(t2, t0),
                   _mm_movelh_ps(t1, t3),
                   _mm_movehl_ps(t3, t1));
   }
   static float Determinant(const Mat4x4& a)
   {
     __m128 t0 = _mm_mul_ps(_mm_mul_ps(SWZ_YXXX(a.r[1]), SWZ_ZZYY(a.r[2])), SWZ_WWWZ(a.r[3]));
     __m128 t1 = _mm_mul_ps(_mm_mul_ps(SWZ_ZZYY(a.r[1]), SWZ_WWWZ(a.r[2])), SWZ_YXXX(a.r[3]));
     __m128 t2 = _mm_mul_ps(_mm_mul_ps(SWZ_WWWZ(a.r[1]), SWZ_YXXX(a.r[2])), SWZ_ZZYY(a.r[3]));
     __m128 t3 = _mm_mul_ps(_mm_mul_ps(SWZ_ZZYY(a.r[1]), SWZ_YXXX(a.r[2])), SWZ_WWWZ(a.r[3]));
     __m128 t4 = _mm_mul_ps(_mm_mul_ps(SWZ_YXXX(a.r[1]), SWZ_WWWZ(a.r[2])), SWZ_ZZYY(a.r[3]));
     __m128 t5 = _mm_mul_ps(_mm_mul_ps(SWZ_WWWZ(a.r[1]), SWZ_ZZYY(a.r[2])), SWZ_YXXX(a.r[3]));
     align16 float f[4];
     _mm_store_ps(f, _mm_mul_ps(a.r[0], _mm_sub_ps(_mm_add_ps(_mm_add_ps(t0, t1), t2), _mm_add_ps(_mm_add_ps(t3, t4), t5))));
     return f[0] - f[1] + f[2] - f[3];
   }
   static const Mat4x4 Inverse(const Mat4x4& a)
   {
     float rd = 1.0f/Determinant(a);
     Mat4x4 m;
     m.e11 = +rd*(a.e22*a.e33*a.e44 + a.e23*a.e34*a.e42 + a.e24*a.e32*a.e43 - a.e24*a.e33*a.e42 - a.e22*a.e34*a.e43 - a.e23*a.e32*a.e44);
     m.e21 = -rd*(a.e21*a.e33*a.e44 + a.e23*a.e34*a.e41 + a.e24*a.e31*a.e43 - a.e24*a.e33*a.e41 - a.e21*a.e34*a.e43 - a.e23*a.e31*a.e44);
     m.e31 = +rd*(a.e21*a.e32*a.e44 + a.e22*a.e34*a.e41 + a.e24*a.e31*a.e42 - a.e24*a.e32*a.e41 - a.e21*a.e34*a.e42 - a.e22*a.e31*a.e44);
     m.e41 = -rd*(a.e21*a.e32*a.e43 + a.e22*a.e33*a.e41 + a.e23*a.e31*a.e42 - a.e23*a.e32*a.e41 - a.e21*a.e33*a.e42 - a.e22*a.e31*a.e43);
     m.e12 = -rd*(a.e12*a.e33*a.e44 + a.e13*a.e34*a.e42 + a.e14*a.e32*a.e43 - a.e14*a.e33*a.e42 - a.e12*a.e34*a.e43 - a.e13*a.e32*a.e44);
     m.e22 = +rd*(a.e11*a.e33*a.e44 + a.e13*a.e34*a.e41 + a.e14*a.e31*a.e43 - a.e14*a.e33*a.e41 - a.e11*a.e34*a.e43 - a.e13*a.e31*a.e44);
     m.e32 = -rd*(a.e11*a.e32*a.e44 + a.e12*a.e34*a.e41 + a.e14*a.e31*a.e42 - a.e14*a.e32*a.e41 - a.e11*a.e34*a.e42 - a.e12*a.e31*a.e44);
     m.e42 = +rd*(a.e11*a.e32*a.e43 + a.e12*a.e33*a.e41 + a.e13*a.e31*a.e42 - a.e13*a.e32*a.e41 - a.e11*a.e33*a.e42 - a.e12*a.e31*a.e43);
     m.e13 = +rd*(a.e12*a.e23*a.e44 + a.e13*a.e24*a.e42 + a.e14*a.e22*a.e43 - a.e14*a.e23*a.e42 - a.e12*a.e24*a.e43 - a.e13*a.e22*a.e44);
     m.e23 = -rd*(a.e11*a.e23*a.e44 + a.e13*a.e24*a.e41 + a.e14*a.e21*a.e43 - a.e14*a.e23*a.e41 - a.e11*a.e24*a.e43 - a.e13*a.e21*a.e44);
     m.e33 = +rd*(a.e11*a.e22*a.e44 + a.e12*a.e24*a.e41 + a.e14*a.e21*a.e42 - a.e14*a.e22*a.e41 - a.e11*a.e24*a.e42 - a.e12*a.e21*a.e44);
     m.e43 = -rd*(a.e11*a.e22*a.e43 + a.e12*a.e23*a.e41 + a.e13*a.e21*a.e42 - a.e13*a.e22*a.e41 - a.e11*a.e23*a.e42 - a.e12*a.e21*a.e43);
     m.e14 = -rd*(a.e12*a.e23*a.e34 + a.e13*a.e24*a.e32 + a.e14*a.e22*a.e33 - a.e14*a.e23*a.e32 - a.e12*a.e24*a.e33 - a.e13*a.e22*a.e34);
     m.e24 = +rd*(a.e11*a.e23*a.e34 + a.e13*a.e24*a.e31 + a.e14*a.e21*a.e33 - a.e14*a.e23*a.e31 - a.e11*a.e24*a.e33 - a.e13*a.e21*a.e34);
     m.e34 = -rd*(a.e11*a.e22*a.e34 + a.e12*a.e24*a.e31 + a.e14*a.e21*a.e32 - a.e14*a.e22*a.e31 - a.e11*a.e24*a.e32 - a.e12*a.e21*a.e34);
     m.e44 = +rd*(a.e11*a.e22*a.e33 + a.e12*a.e23*a.e31 + a.e13*a.e21*a.e32 - a.e13*a.e22*a.e31 - a.e11*a.e23*a.e32 - a.e12*a.e21*a.e33);
     return m;
   }
   static const Mat4x4 LookAtD3D(const Vec3& eye, const Vec3& target, const Vec3& up)
   {
     Vec3 vz = Vec3::Normalize(target - eye);
     Vec3 vx = Vec3::Normalize(Vec3::Cross(up, vz));
     Vec3 vy = Vec3::Cross(vz, vx);
     Mat4x4 m = Mat4x4::Transpose(Mat4x4(Vec3::Vector(vx), Vec3::Vector(vy), Vec3::Vector(vz), c_WOne));
     m.e41 = Vec3::Dot(-vx, eye);
     m.e42 = Vec3::Dot(-vy, eye);
     m.e43 = Vec3::Dot(-vz, eye);
     return m;
   }
   static const Mat4x4 OrthoD3D(float l, float r, float b, float t, float zn, float zf)
   {
     Vec3 s = Vec3::Rcp(Vec3(l - r, b - t, zn - zf));
     return ScalingTranslationD3D(s*Vec3::Constant<-2,-2,-1>(), s*Vec3(l + r, t + b, zn));
   }
};

finline const Vec4 operator * (const Vec4& v, const Mat4x4& m)
{
  return _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(SWZ_XXXX(v.r), m.r[0]), _mm_mul_ps(SWZ_YYYY(v.r), m.r[1])), _mm_mul_ps(SWZ_ZZZZ(v.r), m.r[2])), _mm_mul_ps(SWZ_WWWW(v.r), m.r[3]));
}

finline const Vec4 operator * (const Vec3& v, const Mat4x4& m)
{
  return _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(SWZ_XXXX(v.r), m.r[0]), _mm_mul_ps(SWZ_YYYY(v.r), m.r[1])), _mm_mul_ps(SWZ_ZZZZ(v.r), m.r[2])), m.r[3]);
}

finline const Vec4 operator * (const Vec2& v, const Mat4x4& m)
{
  return _mm_add_ps(_mm_add_ps(_mm_mul_ps(SWZ_XXXX(v.r), m.r[0]), _mm_mul_ps(SWZ_YYYY(v.r), m.r[1])), m.r[3]);
}

finline const Mat4x4 operator * (const Mat4x4& a, const Mat4x4& b)
{
  return Mat4x4(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(SWZ_XXXX(a.r[0]), b.r[0]), _mm_mul_ps(SWZ_YYYY(a.r[0]), b.r[1])), _mm_mul_ps(SWZ_ZZZZ(a.r[0]), b.r[2])), _mm_mul_ps(SWZ_WWWW(a.r[0]), b.r[3])),
                _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(SWZ_XXXX(a.r[1]), b.r[0]), _mm_mul_ps(SWZ_YYYY(a.r[1]), b.r[1])), _mm_mul_ps(SWZ_ZZZZ(a.r[1]), b.r[2])), _mm_mul_ps(SWZ_WWWW(a.r[1]), b.r[3])),
                _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(SWZ_XXXX(a.r[2]), b.r[0]), _mm_mul_ps(SWZ_YYYY(a.r[2]), b.r[1])), _mm_mul_ps(SWZ_ZZZZ(a.r[2]), b.r[2])), _mm_mul_ps(SWZ_WWWW(a.r[2]), b.r[3])),
                _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(SWZ_XXXX(a.r[3]), b.r[0]), _mm_mul_ps(SWZ_YYYY(a.r[3]), b.r[1])), _mm_mul_ps(SWZ_ZZZZ(a.r[3]), b.r[2])), _mm_mul_ps(SWZ_WWWW(a.r[3]), b.r[3])));
}

finline const Vec3 Vec3::Project(const Vec3& v, const Mat4x4& m)
{
  __m128 t = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(SWZ_XXXX(v.r), m.r[0]), _mm_mul_ps(SWZ_YYYY(v.r), m.r[1])), _mm_mul_ps(SWZ_ZZZZ(v.r), m.r[2])), m.r[3]);
  return _mm_div_ps(t, SWZ_WWWW(t));
}

finline const Vec2 Vec2::Project(const Vec2& v, const Mat4x4& m)
{
  __m128 t = _mm_add_ps(_mm_add_ps(_mm_mul_ps(SWZ_XXXX(v.r), m.r[0]), _mm_mul_ps(SWZ_YYYY(v.r), m.r[1])), m.r[3]);
  return _mm_div_ps(t, SWZ_WWWW(t));
}

#endif //#ifndef __MAT4X4
