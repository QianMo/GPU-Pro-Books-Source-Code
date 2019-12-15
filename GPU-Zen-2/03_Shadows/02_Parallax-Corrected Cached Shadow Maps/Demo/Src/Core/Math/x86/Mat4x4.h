/* The functions <something>D3D (e.g. OrthoD3D) construct matrices for left 
   multiplication of the matrix with a vector, b = a*M.

   The functions OBB<something> (e.g. OBBSetScalingD3D) work with specific affine 
   transforms, which represent scaling followed by rotation followed by translation.
*/

#ifndef __MAT4X4
#define __MAT4X4

class Quat;

static const IntegerMask c_InvMat_Sign0 = {0, 0x80000000, 0, 0x80000000};
static const IntegerMask c_InvMat_Sign1 = {0x80000000, 0, 0x80000000, 0};

class Mat4x4 : public MathLibObject
{
public:
  union
  {
    struct { float e11, e12, e13, e14, e21, e22, e23, e24, e31, e32, e33, e34, e41, e42, e43, e44; };
    float e[16];
    __m128 r[4];
  };

  finline Mat4x4() { }
  finline Mat4x4(const __m128& a, const __m128& b, const __m128& c, const __m128& d) { r[0] = a; r[1] = b; r[2] = c; r[3] = d; }
  finline Mat4x4(const Mat4x4& a) { r[0] = a.r[0]; r[1] = a.r[1]; r[2] = a.r[2]; r[3] = a.r[3]; }
  finline Mat4x4(const float* a) { r[0] = _mm_loadu_ps(a); r[1] = _mm_loadu_ps(&a[4]); r[2] = _mm_loadu_ps(&a[8]); r[3] = _mm_loadu_ps(&a[12]); }

  finline const Mat4x4& operator = (const Mat4x4& a) { r[0] = a.r[0]; r[1] = a.r[1]; r[2] = a.r[2]; r[3] = a.r[3]; return *this; }

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
  static finline float Determinant(const Mat4x4& a)
  {
    __m128 t0 = _mm_mul_ps(_mm_mul_ps(SWZ_YXXX(a.r[1]), SWZ_ZZYY(a.r[2])), SWZ_WWWZ(a.r[3]));
    __m128 t1 = _mm_mul_ps(_mm_mul_ps(SWZ_ZZYY(a.r[1]), SWZ_WWWZ(a.r[2])), SWZ_YXXX(a.r[3]));
    __m128 t2 = _mm_mul_ps(_mm_mul_ps(SWZ_WWWZ(a.r[1]), SWZ_YXXX(a.r[2])), SWZ_ZZYY(a.r[3]));
    __m128 t3 = _mm_mul_ps(_mm_mul_ps(SWZ_ZZYY(a.r[1]), SWZ_YXXX(a.r[2])), SWZ_WWWZ(a.r[3]));
    __m128 t4 = _mm_mul_ps(_mm_mul_ps(SWZ_YXXX(a.r[1]), SWZ_WWWZ(a.r[2])), SWZ_ZZYY(a.r[3]));
    __m128 t5 = _mm_mul_ps(_mm_mul_ps(SWZ_WWWZ(a.r[1]), SWZ_ZZYY(a.r[2])), SWZ_YXXX(a.r[3]));
    return _mm_cvtss_f32(dp_m128(_mm_xor_ps(a.r[0], c_InvMat_Sign0.f), _mm_sub_ps(_mm_add_ps(_mm_add_ps(t0, t1), t2), _mm_add_ps(_mm_add_ps(t3, t4), t5))));
  }
  static const Mat4x4 Inverse(const Mat4x4& a)
  {
    __m128 t00 = _mm_mul_ps(_mm_mul_ps(SWZ_YXXX(a.r[1]), SWZ_ZZYY(a.r[2])), SWZ_WWWZ(a.r[3]));
    __m128 t01 = _mm_mul_ps(_mm_mul_ps(SWZ_ZZYY(a.r[1]), SWZ_WWWZ(a.r[2])), SWZ_YXXX(a.r[3]));
    __m128 t02 = _mm_mul_ps(_mm_mul_ps(SWZ_WWWZ(a.r[1]), SWZ_YXXX(a.r[2])), SWZ_ZZYY(a.r[3]));
    __m128 t03 = _mm_mul_ps(_mm_mul_ps(SWZ_WWWZ(a.r[1]), SWZ_ZZYY(a.r[2])), SWZ_YXXX(a.r[3]));
    __m128 t04 = _mm_mul_ps(_mm_mul_ps(SWZ_YXXX(a.r[1]), SWZ_WWWZ(a.r[2])), SWZ_ZZYY(a.r[3]));
    __m128 t05 = _mm_mul_ps(_mm_mul_ps(SWZ_ZZYY(a.r[1]), SWZ_YXXX(a.r[2])), SWZ_WWWZ(a.r[3]));
    __m128 t10 = _mm_mul_ps(_mm_mul_ps(SWZ_YXXX(a.r[0]), SWZ_ZZYY(a.r[2])), SWZ_WWWZ(a.r[3]));
    __m128 t11 = _mm_mul_ps(_mm_mul_ps(SWZ_ZZYY(a.r[0]), SWZ_WWWZ(a.r[2])), SWZ_YXXX(a.r[3]));
    __m128 t12 = _mm_mul_ps(_mm_mul_ps(SWZ_WWWZ(a.r[0]), SWZ_YXXX(a.r[2])), SWZ_ZZYY(a.r[3]));
    __m128 t13 = _mm_mul_ps(_mm_mul_ps(SWZ_WWWZ(a.r[0]), SWZ_ZZYY(a.r[2])), SWZ_YXXX(a.r[3]));
    __m128 t14 = _mm_mul_ps(_mm_mul_ps(SWZ_YXXX(a.r[0]), SWZ_WWWZ(a.r[2])), SWZ_ZZYY(a.r[3]));
    __m128 t15 = _mm_mul_ps(_mm_mul_ps(SWZ_ZZYY(a.r[0]), SWZ_YXXX(a.r[2])), SWZ_WWWZ(a.r[3]));
    __m128 t20 = _mm_mul_ps(_mm_mul_ps(SWZ_YXXX(a.r[0]), SWZ_ZZYY(a.r[1])), SWZ_WWWZ(a.r[3]));
    __m128 t21 = _mm_mul_ps(_mm_mul_ps(SWZ_ZZYY(a.r[0]), SWZ_WWWZ(a.r[1])), SWZ_YXXX(a.r[3]));
    __m128 t22 = _mm_mul_ps(_mm_mul_ps(SWZ_WWWZ(a.r[0]), SWZ_YXXX(a.r[1])), SWZ_ZZYY(a.r[3]));
    __m128 t23 = _mm_mul_ps(_mm_mul_ps(SWZ_WWWZ(a.r[0]), SWZ_ZZYY(a.r[1])), SWZ_YXXX(a.r[3]));
    __m128 t24 = _mm_mul_ps(_mm_mul_ps(SWZ_YXXX(a.r[0]), SWZ_WWWZ(a.r[1])), SWZ_ZZYY(a.r[3]));
    __m128 t25 = _mm_mul_ps(_mm_mul_ps(SWZ_ZZYY(a.r[0]), SWZ_YXXX(a.r[1])), SWZ_WWWZ(a.r[3]));
    __m128 t30 = _mm_mul_ps(_mm_mul_ps(SWZ_YXXX(a.r[0]), SWZ_ZZYY(a.r[1])), SWZ_WWWZ(a.r[2]));
    __m128 t31 = _mm_mul_ps(_mm_mul_ps(SWZ_ZZYY(a.r[0]), SWZ_WWWZ(a.r[1])), SWZ_YXXX(a.r[2]));
    __m128 t32 = _mm_mul_ps(_mm_mul_ps(SWZ_WWWZ(a.r[0]), SWZ_YXXX(a.r[1])), SWZ_ZZYY(a.r[2]));
    __m128 t33 = _mm_mul_ps(_mm_mul_ps(SWZ_WWWZ(a.r[0]), SWZ_ZZYY(a.r[1])), SWZ_YXXX(a.r[2]));
    __m128 t34 = _mm_mul_ps(_mm_mul_ps(SWZ_YXXX(a.r[0]), SWZ_WWWZ(a.r[1])), SWZ_ZZYY(a.r[2]));
    __m128 t35 = _mm_mul_ps(_mm_mul_ps(SWZ_ZZYY(a.r[0]), SWZ_YXXX(a.r[1])), SWZ_WWWZ(a.r[2]));
    __m128 r0 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_add_ps(_mm_add_ps(t00, t01), t02), t03), t04), t05);
    __m128 r1 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_add_ps(_mm_add_ps(t10, t11), t12), t13), t14), t15);
    __m128 r2 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_add_ps(_mm_add_ps(t20, t21), t22), t23), t24), t25);
    __m128 r3 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_add_ps(_mm_add_ps(t30, t31), t32), t33), t34), t35);
    Vec4 d = Vec4::Rcp(Vec4(Determinant(a)));
    return Mat4x4::Transpose(Mat4x4(_mm_mul_ps(r0, _mm_xor_ps(d.r, c_InvMat_Sign0.f)),
                                    _mm_mul_ps(r1, _mm_xor_ps(d.r, c_InvMat_Sign1.f)),
                                    _mm_mul_ps(r2, _mm_xor_ps(d.r, c_InvMat_Sign0.f)),
                                    _mm_mul_ps(r3, _mm_xor_ps(d.r, c_InvMat_Sign1.f))));
  }

  static finline const Mat4x4 Mat4x4::Identity();
  static finline const Mat4x4 OrthoD3D(float l, float r, float b, float t, float zn, float zf);
  static finline const Mat4x4 ProjectionD3D(float l, float r, float b, float t, float zn, float zf);
  static finline const Mat4x4 ProjectionD3D(float fov, float aspect, float zn, float zf);
  static finline const Mat4x4 ProjectionGL(float fov, float aspect, float zn, float zf);
  static finline const Mat4x4 LookAtD3D(const Vec3& eye, const Vec3& target, const Vec3& up);
  static finline const Quat AsQuaternion(const Mat4x4& m, float determinantCubeRoot);
  static finline const Mat4x4 OBBSetScalingD3D(const Mat4x4& m, const Vec3& s);
  static finline const Quat AsQuaternion(const Mat4x4& m);
  static finline const Mat4x4 SetTranslationD3D(const Mat4x4& a, const Vec3& b);
  static finline const Mat4x4 Scaling(const Vec3& a);
  static finline const Mat4x4 TranslationD3D(const Vec3& a);
  static finline const Mat4x4 ScalingTranslationD3D(const Vec3& a, const Vec3& b);
  static finline const Mat4x4 OBBInverseD3D(const Mat4x4& SRT);
  static finline void OBBtoAABB_D3D(const Mat4x4& obb, Vec3& AABBMin, Vec3& AABBMax);
  static finline float FrobeniusNorm(const Mat4x4& m);
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

finline bool operator != (const Mat4x4& a, const Mat4x4& b)
{
  return _mm_movemask_ps(_mm_or_ps(_mm_or_ps(_mm_cmpneq_ps(a.r[0], b.r[0]), _mm_cmpneq_ps(a.r[1], b.r[1])),
                                   _mm_or_ps(_mm_cmpneq_ps(a.r[2], b.r[2]), _mm_cmpneq_ps(a.r[3], b.r[3]))))!=0;
}

finline bool operator == (const Mat4x4& a, const Mat4x4& b)
{
  return _mm_movemask_ps(_mm_or_ps(_mm_or_ps(_mm_cmpneq_ps(a.r[0], b.r[0]), _mm_cmpneq_ps(a.r[1], b.r[1])),
                                   _mm_or_ps(_mm_cmpneq_ps(a.r[2], b.r[2]), _mm_cmpneq_ps(a.r[3], b.r[3]))))==0;
}

#endif //#ifndef __MAT4X4
