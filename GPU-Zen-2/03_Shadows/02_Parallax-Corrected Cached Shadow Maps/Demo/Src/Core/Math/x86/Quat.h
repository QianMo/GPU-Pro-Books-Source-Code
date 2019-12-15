#ifndef __QUAT
#define __QUAT

static const IntegerMask c_Conj_Sign  = {0x80000000, 0x80000000, 0x80000000, 0};
static const IntegerMask c_Q2M_Sign0  = {0x80000000, 0, 0x80000000, 0};
static const IntegerMask c_Q2M_Sign1  = {0, 0, 0, 0x80000000};
static const IntegerMask c_QMUL_Sign0 = {0, 0x80000000, 0, 0x80000000};
static const IntegerMask c_QMUL_Sign1 = {0, 0, 0x80000000, 0x80000000};
static const IntegerMask c_QMUL_Sign2 = {0x80000000, 0, 0, 0x80000000};
static const __m128 c_XYZOne = {1, 1, 1, 0};

class Quat : public Vec4
{
public:
  finline Quat()                                     { }
  finline Quat(__m128 a)                             { r = a; }
  finline Quat(const Quat& a)                        { r = a.r; }
  finline Quat(const Vec4& a)                        { r = a.r; }
  finline Quat(const Vec3& v, float a)               { Vec4 sinA, cosA; SinCos(Vec4(a*0.5f), sinA, cosA); r = _mm_mul_ps(v.r, sinA.r); w = _mm_cvtss_f32(cosA.r); }
  finline Quat(float a, float b, float c, float d)   { x = a; y = b; z = c; w = d; }
  finline Quat(const float* a)                       { r = _mm_loadu_ps(a); }

  static finline const Quat Conjugate(const Quat& a) { return _mm_xor_ps(a.r, c_Conj_Sign.f); }

  static finline const Mat4x4 AsMatrixD3D(const Quat& a)
  {
    __m128 t0 = _mm_and_ps(_mm_add_ps(a.r, a.r), c_WMask.f);
    __m128 t1 = _mm_mul_ps(t0, SWZ_YZXW(a.r));
    __m128 t2 = _mm_mul_ps(t0, SWZ_WWWW(a.r));
    __m128 t3 = _mm_mul_ps(t0, a.r);
    __m128 r0 = _mm_add_ps(SWZ_XXZW(t1), _mm_xor_ps(SWZ_ZZYW(t2), c_Q2M_Sign0.f));
    __m128 r2 = _mm_add_ps(SWZ_YWZY(t1), _mm_xor_ps(SWZ_XWYX(t2), c_Q2M_Sign1.f));
    __m128 r1 = _mm_sub_ps(c_XYZOne, _mm_add_ps(SWZ_YXXW(t3), SWZ_ZZYW(t3)));
    return Mat4x4(_mm_move_ss(r0, r1),
                  _mm_movelh_ps(_mm_move_ss(r1, r0), r2),
                  _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(3,2,3,2)),
                  c_WOne);
  }
  static finline const Vec3 Transform(const Vec3& v, const Quat& q)
  {
    __m128 t = _mm_mul_ps(SWZ_YZXW(q.r), SWZ_ZXYW(v.r));
    t = _mm_sub_ps(t, _mm_mul_ps(SWZ_ZXYW(q.r), SWZ_YZXW(v.r)));
    t = _mm_add_ps(t, t);
    __m128 r = _mm_add_ps(v.r, _mm_mul_ps(t, SWZ_WWWW(q.r)));
    r = _mm_add_ps(r, _mm_mul_ps(SWZ_YZXW(q.r), SWZ_ZXYW(t)));
    r = _mm_sub_ps(r, _mm_mul_ps(SWZ_ZXYW(q.r), SWZ_YZXW(t)));
    return r;
  }
  static finline const Quat Multiply(const Quat& b, const Quat& a)
  {
    __m128 t0 = _mm_xor_ps(_mm_mul_ps(SWZ_XXXX(a.r), SWZ_WZYX(b.r)), c_QMUL_Sign0.f);
    __m128 t1 = _mm_xor_ps(_mm_mul_ps(SWZ_YYYY(a.r), SWZ_ZWXY(b.r)), c_QMUL_Sign1.f);
    __m128 t2 = _mm_xor_ps(_mm_mul_ps(SWZ_ZZZZ(a.r), SWZ_YXWZ(b.r)), c_QMUL_Sign2.f);
    __m128 t3 = _mm_mul_ps(SWZ_WWWW(a.r), b.r);
    return _mm_add_ps(t0, _mm_add_ps(t1, _mm_add_ps(t2, t3)));
  }
};

static const IntegerMask c_RMatQuat_Sign0  = {0, 0x80000000, 0x80000000, 0};
static const IntegerMask c_RMatQuat_Sign1  = {0x80000000, 0, 0x80000000, 0};
static const IntegerMask c_RMatQuat_Sign2  = {0x80000000, 0x80000000, 0, 0};
static const IntegerMask c_RMatQuat_Sign3  = {0, 0x80000000, 0, 0x80000000};

finline const Quat Mat4x4::AsQuaternion(const Mat4x4& m, float determinantCubeRoot)
{
  __m128 t0 = _mm_add_ps(_mm_xor_ps(SWZ_XXXX(m.r[0]), c_RMatQuat_Sign0.f), _mm_set1_ps(determinantCubeRoot));
  __m128 t1 = _mm_add_ps(_mm_xor_ps(SWZ_YYYY(m.r[1]), c_RMatQuat_Sign1.f), t0);
  __m128 t2 = _mm_add_ps(_mm_xor_ps(SWZ_ZZZZ(m.r[2]), c_RMatQuat_Sign2.f), t1);
  __m128 t3 = _mm_mul_ps(_mm_set1_ps(0.5f), _mm_sqrt_ps(_mm_max_ps(t2, _mm_setzero_ps())));
  __m128 t4 = _mm_shuffle_ps(m.r[0], m.r[1], _MM_SHUFFLE(3,2,2,1));
  __m128 t5 = _mm_shuffle_ps(m.r[2], m.r[1], _MM_SHUFFLE(3,0,0,1));
  __m128 t6 = _mm_add_ps(_mm_xor_ps(SWZ_ZYXW(t4), c_RMatQuat_Sign3.f), _mm_xor_ps(t5, c_RMatQuat_Sign1.f));
  return _mm_or_ps(_mm_andnot_ps(c_SignMask.f, t3), _mm_and_ps(c_SignMask.f, t6));
}

#endif //#ifndef __QUAT
