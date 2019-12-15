#ifndef __MAT4X4d
#define __MAT4X4d

static const IntegerMask c_InvMat_Sign0D = {0, 0, 0, 0x80000000};
static const IntegerMask c_InvMat_Sign1D = {0, 0x80000000, 0, 0};

class Mat4x4d : public MathLibObject
{
public:
  union
  {
    struct { double e11, e12, e13, e14, e21, e22, e23, e24, e31, e32, e33, e34, e41, e42, e43, e44; };
    double e[16];
    Vec4d_POD r[4];
  };

  finline Mat4x4d() { }
  finline Mat4x4d(const Vec4d& a, const Vec4d& b, const Vec4d& c, const Vec4d& d) { r[0] = a; r[1] = b; r[2] = c; r[3] = d; }
  finline Mat4x4d(const Mat4x4d& a) { r[0] = a.r[0]; r[1] = a.r[1]; r[2] = a.r[2]; r[3] = a.r[3]; }
  finline Mat4x4d(const double* a) { r[0] = Vec4d(a); r[1] = Vec4d(&a[4]); r[2] = Vec4d(&a[8]); r[3] = Vec4d(&a[12]); }

  finline const Mat4x4d& operator = (const Mat4x4d& a) { r[0] = a.r[0]; r[1] = a.r[1]; r[2] = a.r[2]; r[3] = a.r[3]; return *this; }

  static finline const Mat4x4 Convert(const Mat4x4d& a) { return Mat4x4 (Vec4d::Convert(a.r[0]), Vec4d::Convert(a.r[1]), Vec4d::Convert(a.r[2]), Vec4d::Convert(a.r[3])); }
  static finline const Mat4x4d Convert(const Mat4x4& a) { return Mat4x4d(Vec4d::Convert(a.r[0]), Vec4d::Convert(a.r[1]), Vec4d::Convert(a.r[2]), Vec4d::Convert(a.r[3])); }

  static finline const Mat4x4d Identity()
  {
    __m128d t00 = _mm_setzero_pd();
    __m128d t10 = _mm_set_sd(1.0);
    __m128d t01 = SWZ_YX(t10);
    return Mat4x4d(Vec4d(t10, t00), Vec4d(t01, t00), Vec4d(t00, t10), Vec4d(t00, t01));
  }
  static finline const Mat4x4d Transpose(const Mat4x4d& a)
  {
    return Mat4x4d(Vec4d(_mm_shuffle_pd(a.r[0].r[0], a.r[1].r[0], _MM_SHUFFLE2(0, 0)), _mm_shuffle_pd(a.r[2].r[0], a.r[3].r[0], _MM_SHUFFLE2(0, 0))),
                   Vec4d(_mm_shuffle_pd(a.r[0].r[0], a.r[1].r[0], _MM_SHUFFLE2(1, 1)), _mm_shuffle_pd(a.r[2].r[0], a.r[3].r[0], _MM_SHUFFLE2(1, 1))),
                   Vec4d(_mm_shuffle_pd(a.r[0].r[1], a.r[1].r[1], _MM_SHUFFLE2(0, 0)), _mm_shuffle_pd(a.r[2].r[1], a.r[3].r[1], _MM_SHUFFLE2(0, 0))),
                   Vec4d(_mm_shuffle_pd(a.r[0].r[1], a.r[1].r[1], _MM_SHUFFLE2(1, 1)), _mm_shuffle_pd(a.r[2].r[1], a.r[3].r[1], _MM_SHUFFLE2(1, 1))));
  }
  static finline double Determinant(const Mat4x4d& a)
  {
    // This function is a plain copy from determinant(dmat4) of glsl-sse2 library (https://github.com/LiraNuna/glsl-sse2)
    __m128d r1 = _mm_mul_pd(a.r[0].r[0], SWZ_YX(a.r[1].r[0]));
    __m128d r2 = _mm_mul_pd(a.r[0].r[1], SWZ_YX(a.r[1].r[1]));
    __m128d r3 = _mm_mul_pd(a.r[2].r[0], SWZ_YX(a.r[3].r[0]));
    __m128d r4 = _mm_mul_pd(a.r[2].r[1], SWZ_YX(a.r[3].r[1]));
    __m128d c1 = _mm_sub_pd(_mm_mul_pd(a.r[2].r[0], SWZ_YY(a.r[3].r[1])), _mm_mul_pd(a.r[3].r[0], SWZ_YY(a.r[2].r[1])));
    __m128d c2 = _mm_sub_pd(_mm_mul_pd(a.r[3].r[0], SWZ_XX(a.r[2].r[1])), _mm_mul_pd(a.r[2].r[0], SWZ_XX(a.r[3].r[1])));
    __m128d r5 = _mm_mul_pd(_mm_sub_pd(_mm_mul_pd(a.r[0].r[1], SWZ_YY(a.r[1].r[0])), _mm_mul_pd(a.r[1].r[1], SWZ_YY(a.r[0].r[0]))), _mm_unpacklo_pd(c1, c2));
    __m128d r6 = _mm_mul_pd(_mm_sub_pd(_mm_mul_pd(a.r[1].r[1], SWZ_XX(a.r[0].r[0])), _mm_mul_pd(a.r[0].r[1], SWZ_XX(a.r[1].r[0]))), _mm_unpackhi_pd(c1, c2));
    __m128d d = _mm_add_pd(r5, r6);
    r1 = _mm_sub_sd(r1, SWZ_YY(r1));
    r2 = _mm_sub_sd(r2, SWZ_YY(r2));
    r3 = _mm_sub_sd(r3, SWZ_YY(r3));
    r4 = _mm_sub_sd(r4, SWZ_YY(r4));
    return _mm_cvtsd_f64(_mm_sub_sd(_mm_add_sd(_mm_mul_sd(r1, r4), _mm_mul_sd(r2, r3)), _mm_add_sd(SWZ_YY(d), d)));
  }
  static finline const Mat4x4d Inverse(const Mat4x4d& a)
  {
    // This function is a plain copy from inverse(dmat4) of glsl-sse2 library (https://github.com/LiraNuna/glsl-sse2)
    __m128d r1 = _mm_mul_pd(a.r[0].r[0], SWZ_YX(a.r[1].r[0]));
    __m128d r2 = _mm_mul_pd(a.r[0].r[1], SWZ_YX(a.r[1].r[1]));
    __m128d r3 = _mm_mul_pd(a.r[2].r[0], SWZ_YX(a.r[3].r[0]));
    __m128d r4 = _mm_mul_pd(a.r[2].r[1], SWZ_YX(a.r[3].r[1]));
    __m128d v11 = _mm_sub_pd(_mm_mul_pd(SWZ_YY(a.r[1].r[0]), a.r[0].r[1]), _mm_mul_pd(SWZ_YY(a.r[0].r[0]), a.r[1].r[1]));
    __m128d v12 = _mm_sub_pd(_mm_mul_pd(SWZ_XX(a.r[0].r[0]), a.r[1].r[1]), _mm_mul_pd(SWZ_XX(a.r[1].r[0]), a.r[0].r[1]));
    __m128d v21 = _mm_sub_pd(_mm_mul_pd(SWZ_YY(a.r[3].r[1]), a.r[2].r[0]), _mm_mul_pd(SWZ_YY(a.r[2].r[1]), a.r[3].r[0]));
    __m128d v22 = _mm_sub_pd(_mm_mul_pd(SWZ_XX(a.r[2].r[1]), a.r[3].r[0]), _mm_mul_pd(SWZ_XX(a.r[3].r[1]), a.r[2].r[0]));
    __m128d d = _mm_add_pd(_mm_mul_pd(v11, _mm_unpacklo_pd(v21, v22)), _mm_mul_pd(v12, _mm_unpackhi_pd(v21, v22)));
    r1 = _mm_sub_sd(r1, SWZ_YY(r1));
    r2 = _mm_sub_sd(r2, SWZ_YY(r2));
    r3 = _mm_sub_sd(r3, SWZ_YY(r3));
    r4 = _mm_sub_sd(r4, SWZ_YY(r4));
    d = _mm_add_sd(SWZ_YY(d), d);
    d = _mm_div_sd(_mm_set_sd(1.0), _mm_sub_sd(_mm_add_sd(_mm_mul_sd(r1, r4), _mm_mul_sd(r2, r3)), d));
    d = SWZ_XX(d);
    __m128d i11 = _mm_sub_pd(_mm_mul_pd(a.r[0].r[0], SWZ_XX(r4)), _mm_add_pd(_mm_mul_pd(v21, SWZ_XX(a.r[0].r[1])), _mm_mul_pd(v22, SWZ_YY(a.r[0].r[1]))));
    __m128d i12 = _mm_sub_pd(_mm_mul_pd(a.r[1].r[0], SWZ_XX(r4)), _mm_add_pd(_mm_mul_pd(v21, SWZ_XX(a.r[1].r[1])), _mm_mul_pd(v22, SWZ_YY(a.r[1].r[1]))));
    __m128d i41 = _mm_sub_pd(_mm_mul_pd(a.r[2].r[1], SWZ_XX(r1)), _mm_add_pd(_mm_mul_pd(v11, SWZ_XX(a.r[2].r[0])), _mm_mul_pd(v12, SWZ_YY(a.r[2].r[0]))));
    __m128d i42 = _mm_sub_pd(_mm_mul_pd(a.r[3].r[1], SWZ_XX(r1)), _mm_add_pd(_mm_mul_pd(v11, SWZ_XX(a.r[3].r[0])), _mm_mul_pd(v12, SWZ_YY(a.r[3].r[0]))));
    __m128d i21 = _mm_sub_pd(_mm_mul_pd(a.r[2].r[0], SWZ_XX(r2)), _mm_sub_pd(_mm_mul_pd(_mm_shuffle_pd(v12, v11, 0x01), a.r[2].r[1]), _mm_mul_pd(SWZ_YX(a.r[2].r[1]), _mm_shuffle_pd(v12, v11, 0x02))));
    __m128d i22 = _mm_sub_pd(_mm_mul_pd(a.r[3].r[0], SWZ_XX(r2)), _mm_sub_pd(_mm_mul_pd(_mm_shuffle_pd(v12, v11, 0x01), a.r[3].r[1]), _mm_mul_pd(SWZ_YX(a.r[3].r[1]), _mm_shuffle_pd(v12, v11, 0x02))));
    __m128d i31 = _mm_sub_pd(_mm_mul_pd(a.r[0].r[1], SWZ_XX(r3)), _mm_sub_pd(_mm_mul_pd(_mm_shuffle_pd(v22, v21, 0x01), a.r[0].r[0]), _mm_mul_pd(SWZ_YX(a.r[0].r[0]), _mm_shuffle_pd(v22, v21, 0x02))));
    __m128d i32 = _mm_sub_pd(_mm_mul_pd(a.r[1].r[1], SWZ_XX(r3)), _mm_sub_pd(_mm_mul_pd(_mm_shuffle_pd(v22, v21, 0x01), a.r[1].r[0]), _mm_mul_pd(SWZ_YX(a.r[1].r[0]), _mm_shuffle_pd(v22, v21, 0x02))));
    __m128d d1 = _mm_xor_pd(d, c_InvMat_Sign0D.d);
    __m128d d2 = _mm_xor_pd(d, c_InvMat_Sign1D.d);
    return Mat4x4d(Vec4d(_mm_mul_pd(_mm_unpackhi_pd(i12, i11), d1), _mm_mul_pd(_mm_unpackhi_pd(i22, i21), d1)),
                   Vec4d(_mm_mul_pd(_mm_unpacklo_pd(i12, i11), d2), _mm_mul_pd(_mm_unpacklo_pd(i22, i21), d2)),
                   Vec4d(_mm_mul_pd(_mm_unpackhi_pd(i32, i31), d1), _mm_mul_pd(_mm_unpackhi_pd(i42, i41), d1)),
                   Vec4d(_mm_mul_pd(_mm_unpacklo_pd(i32, i31), d2), _mm_mul_pd(_mm_unpacklo_pd(i42, i41), d2)));
  }
  static finline const Mat4x4d ScalingTranslationD3D(const Vec3d& s, const Vec3d& t)
  {
     return Mat4x4d(Vec4d(_mm_unpacklo_pd(s.r[0], _mm_setzero_pd()), _mm_setzero_pd()),
                    Vec4d(_mm_unpackhi_pd(_mm_setzero_pd(), s.r[0]), _mm_setzero_pd()),
                    Vec4d(_mm_setzero_pd(), s.r[1]),
                    Vec3d::Point(t));
  }

  static finline const Mat4x4d LookAtD3D(const Vec3d& eye, const Vec3d& target, const Vec3d& up);
  static finline const Mat4x4d OrthoD3D(double l, double r, double b, double t, double zn, double zf);
  static finline const Mat4x4d SetTranslationD3D(const Mat4x4d& a, const Vec3d& b);
  static finline const Mat4x4d TranslationD3D(const Vec3d& a);
};

#endif //#ifndef __MAT4X4d
