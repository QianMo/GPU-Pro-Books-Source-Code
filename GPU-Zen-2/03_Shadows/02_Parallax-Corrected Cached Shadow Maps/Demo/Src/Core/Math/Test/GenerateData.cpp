#include <d3dx9.h>
#include <stdio.h>
#include "Util.h"
#include "Config.h"

Mat4x4 RandomMatrix(RandomLCG& r)
{
  Quat q = Quat::Normalize(RandomVec4(r));
  return Mat4x4::Scaling(RandomVec3(r))*Quat::AsMatrixD3D(q)*Mat4x4::TranslationD3D(RandomVec3(r));
}

void SaveMatrix(const char* pszFile, const Mat4x4& m)
{
  FILE* pOut = fopen(pszFile, "w+");
  for(int i=0; i<16; ++i)
    fprintf(pOut, "%.12f ", m.e[i]);
  fclose(pOut);
}

void GenerateMat4x4()
{
  RandomLCG r(-3.0f, 3.0f);
  for(int i=0; i<NUMBER_OF_TEST_MATRICES; ++i)
  {
    Mat4x4 m = RandomMatrix(r);
    SaveMatrix(GetFmtFileName("matrix_%d.dat", i), m);
    D3DXMATRIX d3dxMat(m.e), d3dxInvMat;
    D3DXMatrixInverse(&d3dxInvMat, NULL, &d3dxMat);
    SaveMatrix(GetFmtFileName("inv_matrix_%d.dat", i), Mat4x4(&d3dxInvMat._11));
  }
}

void SaveVector4(const char* pszFile, const Vec4& v)
{
  FILE* pOut = fopen(pszFile, "w+");
  for(int i=0; i<4; ++i)
    fprintf(pOut, "%.12f ", v[i]);
  fclose(pOut);
}

void GenerateQuaternion()
{
  RandomLCG r(-3.0f, 3.0f);
  for(int i=0; i<NUMBER_OF_TEST_QUATERNIONS; ++i)
  {
    Vec4 _a = RandomVec4(r); D3DXQUATERNION a(&_a.x);
    D3DXQUATERNION na; D3DXQuaternionNormalize(&na, &a);
    Vec4 _b = RandomVec4(r); D3DXQUATERNION b(&_b.x);
    D3DXQUATERNION nb; D3DXQuaternionNormalize(&nb, &b);
    D3DXQUATERNION ab;
    D3DXQuaternionMultiply(&ab, &na, &nb);
    SaveVector4(GetFmtFileName("q_a_%d.dat", i),  _a);
    SaveVector4(GetFmtFileName("q_b_%d.dat", i),  _b);
    SaveVector4(GetFmtFileName("q_ab_%d.dat", i), Quat(&ab.x));
    D3DXMATRIX ma; D3DXMatrixRotationQuaternion(&ma, &na);
    SaveMatrix(GetFmtFileName("q_ma_%d.dat", i), Mat4x4(&ma._11));
    D3DXMATRIX mb; D3DXMatrixRotationQuaternion(&mb, &nb);
    SaveMatrix(GetFmtFileName("q_mb_%d.dat", i), Mat4x4(&mb._11));
  }
}

int main()
{
  GenerateMat4x4();
  GenerateQuaternion();
  printf("Done.\n");
  return 0;
}
