/* Frustum culling.

   The culling is accurate, IsIntersecting() does not produce false 
   positives, but it can be expensive in certain presumably rare cases.

   IsIntersecting(const Mat4x4& OBB, float BSphereRadius) is the fastest 
   codepath given that BSphereRadius is *precomputed* (call GetBSphereRadius
   to obtain one). If you don't have precomputed radius available, use 
   IsIntersecting(const Mat4x4& OBB) instead. */

#ifndef __FRUSTUM
#define __FRUSTUM

inline float GetBSphereRadius(const Mat4x4& OBB)
{
  return Vec4::Length(Vec3::Vector(Vec3(1))*OBB);
}

static const IntegerMask c_BBoxCornerSign[] =
{
  {0x80000000, 0x80000000, 0x80000000, 0},
  {0x00000000, 0x80000000, 0x80000000, 0},
  {0x00000000, 0x00000000, 0x80000000, 0},
  {0x80000000, 0x00000000, 0x80000000, 0},
  {0x80000000, 0x00000000, 0x00000000, 0},
  {0x80000000, 0x80000000, 0x00000000, 0},
  {0x00000000, 0x80000000, 0x00000000, 0},
  {0x00000000, 0x00000000, 0x00000000, 0},
};

finline Vec4 TransformBoxCorner(int i, const Mat4x4& m)
{
  const Vec4 v = c_BBoxCornerSign[i].f;
  return (Vec4::Swizzle<x,x,x,x>(v) ^ Vec4(m.r[0])) +
         (Vec4::Swizzle<y,y,y,y>(v) ^ Vec4(m.r[1])) +
         (Vec4::Swizzle<z,z,z,z>(v) ^ Vec4(m.r[2])) +
         Vec4(m.r[3]);
}

static const int c_BBoxFaceIndices[6][4]=
{
  {0,1,2,3},
  {0,5,6,1},
  {5,4,7,6},
  {4,3,2,7},
  {1,6,7,2},
  {3,4,5,0},
};

static const Mat4x4 c_D3DZ_to_OGLZ(Vec4(1,0,0,0), Vec4(0,1,0,0), Vec4(0,0,2,0), Vec4(0,0,-1,1));
static const Mat4x4 c_OGLZ_to_D3DZ(Vec4(1,0,0,0), Vec4(0,1,0,0), Vec4(0,0,0.5f,0), Vec4(0,0,0.5f,1));
static const Vec4 c_NZZW(0, 0, FLT_MAX, FLT_MAX);

class Frustum : public MathLibObject
{
public:
  static const Frustum FromViewProjectionMatrixD3D(const Mat4x4& MVP, const Mat4x4& InvMVP)
  {
    Frustum f;
    f.m_Planes03.r[0] = PlaneNormalize(Vec4(MVP.e14 + MVP.e11, MVP.e24 + MVP.e21, MVP.e34 + MVP.e31, MVP.e44 + MVP.e41));
    f.m_Planes03.r[1] = PlaneNormalize(Vec4(MVP.e14 - MVP.e11, MVP.e24 - MVP.e21, MVP.e34 - MVP.e31, MVP.e44 - MVP.e41));
    f.m_Planes03.r[2] = PlaneNormalize(Vec4(MVP.e14 - MVP.e12, MVP.e24 - MVP.e22, MVP.e34 - MVP.e32, MVP.e44 - MVP.e42));
    f.m_Planes03.r[3] = PlaneNormalize(Vec4(MVP.e14 + MVP.e12, MVP.e24 + MVP.e22, MVP.e34 + MVP.e32, MVP.e44 + MVP.e42));
    f.m_Planes45.r[0] = PlaneNormalize(Vec4(MVP.e13,           MVP.e23,           MVP.e33,           MVP.e43          ));
    f.m_Planes45.r[1] = PlaneNormalize(Vec4(MVP.e14 - MVP.e13, MVP.e24 - MVP.e23, MVP.e34 - MVP.e33, MVP.e44 - MVP.e43));
    f.m_Planes45.r[2] = Vec4::Zero();
    f.m_Planes45.r[3] = Vec4::Zero();
    f.m_Planes03 = Mat4x4::Transpose(f.m_Planes03);
    f.m_Planes45 = Mat4x4::Transpose(f.m_Planes45);
    f.m_MVP = MVP*c_D3DZ_to_OGLZ;
    f.m_InvMVP = c_OGLZ_to_D3DZ*InvMVP;
    return f;
  }
  static const Frustum FromViewProjectionMatrixD3D(const Mat4x4& MVP)
  {
    return FromViewProjectionMatrixD3D(MVP, Mat4x4::Inverse(MVP));
  }
  inline bool IsIntersecting(const Mat4x4& OBB, float BSphereRadius)
  {
    Vec4 d03 = Vec4(OBB.r[3])*m_Planes03;
    Vec4 d45 = Vec4(OBB.r[3])*m_Planes45;

    // is OBB's bounding sphere located below any plane?
    Vec4 n = Vec4(-BSphereRadius);
    if(!(d03>=n) | !(d45>=n))
      return false;

    // is OBB's bounding sphere fully inside frustum?
    Vec4 p = Vec4(BSphereRadius);
    if((d03>=p) & ((d45 | c_NZZW)>=p))
      return true;

    return IsIntersecting(OBB);
  }
  inline bool IsIntersecting(const Mat4x4& OBB)
  {
    Mat4x4 OBBP03 = OBB*m_Planes03;
    Mat4x4 OBBP45 = OBB*m_Planes45;
    Vec4 a03(-1);
    Vec4 a45(-1);
    for(int i=0; i<8; ++i)
    {
      Vec4 s03 = TransformBoxCorner(i, OBBP03);
      Vec4 s45 = TransformBoxCorner(i, OBBP45);

      // is OBB corner inside frustum?
      if((((s03 | s45) & c_SignMask.f) | Vec4(1.0f))>=Vec4::Zero()) 
        return true;

      a03 = (a03 & s03);
      a45 = (a45 & s45);
    }
    // are all OBB's corners located below any plane?
    if(!((((a03 | a45) & c_SignMask.f) | Vec4(1.0f))>=Vec4::Zero()))
      return false;

    Mat4x4 iMVPiOBB = m_InvMVP*Mat4x4::OBBInverseD3D(OBB);
    for(int i=0; i<8; ++i)
    {
      // is frustum's corner inside OBB?
      Vec4 v4 = TransformBoxCorner(i, iMVPiOBB);
      if((Vec3(v4)>=Vec3(-v4.w)) & (Vec3(v4)<=Vec3(v4.w)))
        return true;
    }

    // at this point the only possibility for OBB to be visible is iff any 
    // OBB's face intersects frustum, thus the most expensive check follows
    Vec4 VB[8];
    Mat4x4 OBBMVP = OBB*m_MVP;
    for(int i=0; i<8; ++i)
      VB[i] = TransformBoxCorner(i, OBBMVP);
    for(int i=0; i<6; ++i)
    {
      Vec4 Face[256];
      for(int j=0; j<4; j++)
        Face[j] = VB[c_BBoxFaceIndices[i][j]];
      if(ClipPolygon(4, Face))
        return true;
    }
    return false;
  }

protected:
  Mat4x4 m_Planes03;
  Mat4x4 m_Planes45;
  Mat4x4 m_MVP, m_InvMVP;

  static int ClipPolygon(int n, Vec4 *v1)
  {
    Vec4 VB0[256];
    Vec4 VB1[256];
    const Vec4 *v = v1;
    int j, index[2]; float d[2];
    for(int i=0; i<3; i++)
    {
      for(int z=-1; z<2; z+=2)
      {
        float f = (float)z;
        index[1] = n-1;
        d[1] = f*v[index[1]][i] + v[index[1]].w;
        for(j=0; j<n; j++)
        {
          index[0] = index[1];
          index[1] = j;
          d[0] = d[1];
          d[1] = f*v[index[1]][i] + v[index[1]].w;
          if(d[1]>0 && d[0]<0) break;
        }
        if(j<n)
        {
          int k = 0;
          Vec4 *tmp = (v==VB0) ? VB1 : VB0;
          tmp[k++] = Vec4::Lerp(v[index[1]], v[index[0]], d[1]/(d[1] - d[0]));
          do
          {
            index[0] = index[1];
            index[1] = (index[1]+1)%n;
            d[0] = d[1];
            d[1] = f*v[index[1]][i] + v[index[1]].w;
            tmp[k++] = v[index[0]];
          } while(d[1]>0);
          tmp[k++] = Vec4::Lerp(v[index[1]], v[index[0]], d[1]/(d[1] - d[0]));
          n = k;
          v = tmp;
        }
        else
        {
          if(d[1]<0) return 0;
        }
      }
    }
    memcpy(v1, v, n*sizeof(v[0]));
    return n;
  }
  static finline Vec4 PlaneNormalize(const Vec4& v)
  {
    return v/Vec3::Length(Vec3(v));
  }
};

#endif//#ifndef __FRUSTUM
