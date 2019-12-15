#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "Util.h"
#include "Config.h"
#define _ASSERT ASSERT
#include "../IO.h"
#include "../../Util/MemoryBuffer.h"

#if defined(ENABLE_SSE4) && !defined(ENABLE_SSE2)
  #error ENABLE_SSE4 should be used together with ENABLE_SSE2.
#endif

template<class T, unsigned N> const T LoadFromFile(const char* pszFile)
{
  MemoryBuffer buf;
  bool bOK = buf.Load(pszFile);
  ASSERT(bOK);
  buf.Seek(buf.Size());
  buf.Write((char)0);
  return MathIO::Read<T, N>(buf.Ptr<char>(0));
}

void TestMat4x4()
{
  for(int i=0; i<NUMBER_OF_TEST_MATRICES; ++i)
  {
    Mat4x4 m = LoadFromFile<Mat4x4, 16>(GetFmtFileName("matrix_%d.dat", i));
    Mat4x4 im = LoadFromFile<Mat4x4, 16>(GetFmtFileName("inv_matrix_%d.dat", i));
    ASSERT(IsEqual(im, Mat4x4::Inverse(m), 1e-5f));
    ASSERT(IsEqual(im, Mat4x4::OBBInverseD3D(m), 1e-5f));
    ASSERT(IsEqual(Mat4x4::Identity(), im*m, 1e-5f));
#ifdef ENABLE_SSE2
    ASSERT(IsEqual(im, Mat4x4d::Convert(Mat4x4d::Inverse(Mat4x4d::Convert(m))), 1e-5f));
    ASSERT(IsEqual(Mat4x4::Identity(), Mat4x4d::Convert(Mat4x4d::Convert(im)*Mat4x4d::Convert(m)), 1e-5f));
    ASSERT(fabsf(Mat4x4::Determinant(m) - Mat4x4d::Determinant(Mat4x4d::Convert(m)))<=1e-5f);
#endif
  }
}

Vec4 RefCbrt(const Vec4& a) { return Vec4(powf(a.x, 1.0f/3.0f), powf(a.y, 1.0f/3.0f), powf(a.z, 1.0f/3.0f), powf(a.w, 1.0f/3.0f)); }
Vec4 RefSin(const Vec4& a) { return Vec4(sinf(a.x), sinf(a.y), sinf(a.z), sinf(a.w)); }
Vec4 RefCos(const Vec4& a) { return Vec4(cosf(a.x), cosf(a.y), cosf(a.z), cosf(a.w)); }

void TestVec4()
{
  RandomLCG r(-2.0f, 2.0f);
  for(int i=0; i<1000; ++i)
  {
    Vec4 a = RandomVec4(r);
    Vec4 b = RandomVec4(r);
    ASSERT(fabsf(a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w - Vec4::Dot(a, b))<=1e-5f);
    ASSERT(fabsf(a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w - Vec4::LengthSq(a))<=1e-5f);
    ASSERT(fabsf(sqrtf(a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w) - Vec4::Length(a))<=1e-5f);
    ASSERT(Vec4::Abs(a/Vec4::Length(a) - Vec4::Normalize(a))<=Vec4(1e-5f));
    ASSERT(Vec4::Abs(Vec4(1.0f)/Vec4::Sqrt(Vec4::Abs(a)) - Vec4::Rsqrt(Vec4::Abs(a)))<=Vec4(1e-5f));
    ASSERT(Vec4::Abs(Vec4(1.0f)/a - Vec4::Rcp(a))<=Vec4(1e-4f));
    ASSERT((Vec4::Swizzle<y,w,x,z>(a)==Vec4(a.y, a.w, a.x, a.z)));
    ASSERT((Vec4::Shuffle<y,w,x,z>(a, b)==Vec4(a.y, a.w, b.x, b.z)));
#ifdef ENABLE_SSE2
    ASSERT(Vec4::Abs(RefCbrt(Vec4::Abs(a)) - Vec4::Cbrt(Vec4::Abs(a)))<=Vec4(1e-5f));
    ASSERT(Vec4::Abs(RefCbrt(Vec4::Abs(10.0f*b)) - Vec4::Cbrt(Vec4::Abs(10.0f*b)))<=Vec4(1e-5f));
    Vec4 testSin, testCos;
    const float angleScale = 20.0f;
    Vec4::SinCos(a*angleScale, testSin, testCos);
    ASSERT(Vec4::Abs(testSin - RefSin(a*angleScale))<=Vec4(1e-5f));
    ASSERT(Vec4::Abs(testCos - RefCos(a*angleScale))<=Vec4(1e-5f));
    ASSERT(fabs(Vec4::LengthSq(a) - Vec4d::LengthSq(Vec4d::Convert(a)))<=1e-5f);
    ASSERT(fabs(Vec4::Length(a) - Vec4d::Length(Vec4d::Convert(a)))<=1e-5f);
    ASSERT(Vec4::Abs(Vec4::Normalize(a) - Vec4d::Convert(Vec4d::Normalize(Vec4d::Convert(a))))<=Vec4(1e-5f));
    ASSERT(Vec4::Abs(Vec3::Cross(a, b) - Vec4d::Convert(Vec3d::Cross(Vec4d::Convert(a), Vec4d::Convert(b))))<=Vec4(1e-5f));
#endif
  }
}

void TestQuaternion()
{
  for(int i=0; i<NUMBER_OF_TEST_QUATERNIONS; ++i)
  {
    Quat a = LoadFromFile<Quat, 4>(GetFmtFileName("q_a_%d.dat", i));
    Quat b = LoadFromFile<Quat, 4>(GetFmtFileName("q_b_%d.dat", i));
    Quat na = Quat::Normalize(a);
    Quat nb = Quat::Normalize(b);
    Quat ab = Quat::Multiply(na, nb);
    ASSERT(Vec4::Abs(LoadFromFile<Quat, 4>(GetFmtFileName("q_ab_%d.dat", i)) - ab)<=Vec4(1e-5f));
    Mat4x4 ma = LoadFromFile<Mat4x4, 16>(GetFmtFileName("q_ma_%d.dat", i));
    Mat4x4 mb = LoadFromFile<Mat4x4, 16>(GetFmtFileName("q_mb_%d.dat", i));
    ASSERT(IsEqual(ma, Quat::AsMatrixD3D(na), 1e-5f));
    ASSERT(IsEqual(mb, Quat::AsMatrixD3D(nb), 1e-5f));
    Quat cna = Mat4x4::AsQuaternion(Quat::AsMatrixD3D(na), 1.0f);
    ASSERT(Vec4::Abs(na - cna)<=Vec4(1e-5f) || Vec4::Abs(na + cna)<=Vec4(1e-5f));
    Quat cnb = Mat4x4::AsQuaternion(Quat::AsMatrixD3D(nb), 1.0f);
    ASSERT(Vec4::Abs(nb - cnb)<=Vec4(1e-5f) || Vec4::Abs(nb + cnb)<=Vec4(1e-5f));
  }
}

#ifdef ENABLE_SSE2

void TestVec4i()
{
  RandomLCG r;
  for(int i=0; i<1000; ++i)
  {
    Vec4i a = RandomVec4i(r);
    Vec4i b = RandomVec4i(r);
    Vec4i c = (100 + (((Vec4i::Abs(Vec4i::Min(a, b)) - Vec4i::Abs(Vec4i::Max(a, b))) >> 1) << 3)) ^ 5;
    Vec4i refc;
    for(int j=0; j<4; ++j)
      refc[j] = (100 + (((abs(std::min(a[j], b[j])) - abs(std::max(a[j], b[j]))) >> 1) << 3)) ^ 5;
    ASSERT(c==refc);
  }
}

#endif

int main()
{
  TestMat4x4();
  TestVec4();
  TestQuaternion();
#ifdef ENABLE_SSE2
  TestVec4i();
#endif
  return 0;
}
