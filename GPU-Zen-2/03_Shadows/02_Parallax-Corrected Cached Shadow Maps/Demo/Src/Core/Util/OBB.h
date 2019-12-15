/* Oriented bounding box. Constructs initial OBB from a point cloud via 
   PCA, then tries to find a better OBB among the boxes with an edge 
   parallel to an edge of cloud's convex hull. Beware of O(n^2). */

#ifndef __OBB
#define __OBB

class OBB2D : public MathLibObject
{
public:
  Vec2 m_Center;
  Vec2 m_Ext[2];

  static int FindConvexHull(const int nVertices, const Vec2 *pVertices, Vec2 *pHull)
  {
    const float Eps = 1e-5f;
    const float EpsSq = Eps*Eps;
    int LeftmostIndex = 0;
    for(int i=1; i<nVertices; ++i)
    {
      float f = pVertices[LeftmostIndex].x - pVertices[i].x;
      if(fabs(f)<Eps)
      {
        if(pVertices[LeftmostIndex].y>pVertices[i].y)
          LeftmostIndex = i;
      }
      else if(f>0)
        LeftmostIndex = i;
    }
    Vec2 Dir0(0, -1);
    int HullSize = 0;
    int Index0 = LeftmostIndex;
    do
    {
      float MaxCos = -FLT_MAX;
      int Index1 = -1;
      Vec2 Dir1;
      for(int j=1; j<nVertices; ++j)
      {
        int k = (Index0 + j)%nVertices;
        Vec2 v = pVertices[k] - pVertices[Index0];
        float l = Vec2::LengthSq(v);
        if(l>EpsSq)
        {
          Vec2 d = Vec2::Normalize(v);
          float f = Vec2::Dot(d, Dir0);
          if(MaxCos<f)
          {
            MaxCos = f;
            Index1 = k;
            Dir1 = d;
          }
        }
      }
      if(Index1<0 || HullSize>=nVertices)
      {
        _ASSERT(!"epic fail");
        return 0;
      }
      pHull[HullSize++] = pVertices[Index1];
      Index0 = Index1;
      Dir0 = Dir1;
    } while(Vec2::LengthSq(pVertices[Index0] - pVertices[LeftmostIndex])>EpsSq);
    return HullSize;
  }
  static OBB2D Create(const int nVertices, const Vec2 *pVertices)
  {
    Vec2 centroid = pVertices[0];
    for(int i=1; i<nVertices; ++i)
      centroid += pVertices[i];
    centroid /= float(nVertices);

    float CovMat[4] = { };
    for(int i=0; i<nVertices; ++i)
    {
      const Vec2& v = pVertices[i] - centroid;
      CovMat[0] += v.x*v.x; CovMat[1] += v.y*v.x;
      CovMat[2] += v.y*v.x; CovMat[3] += v.y*v.y;
    }
    float b = -(CovMat[0] + CovMat[3]);
    float c = CovMat[0]*CovMat[3] - CovMat[1]*CovMat[2];
    float f = b*b - 4.0f*c;
    float d = f>0 ? sqrt(f) : 0;
    float L1 = (-b + d)/2.0f;
    float L2 = (-b - d)/2.0f;
    float Lm = std::max(L1, L2);
    const float kEps = 1e-5f;
    Vec2 Ed = Vec2::Normalize( fabsf( CovMat[2] ) > kEps ? Vec2( Lm - CovMat[3], CovMat[2] ) : Vec2( 1.0f, 0.0f ) );

    Vec2* pFrustumHull = (Vec2*)alloca(sizeof(Vec2)*nVertices);
    int nHullSize = FindConvexHull(nVertices, pVertices, pFrustumHull);

    OBB2D BBox;
    float BestOBBArea = FLT_MAX;
    Vec2 TestDir = Ed;
    for(int i=0; i<=nHullSize; ++i)
    {
      Mat4x4 InvRotLS = Mat4x4::Identity();
      InvRotLS.r[0] = TestDir;
      InvRotLS.r[1] = Vec3::Cross(c_ZAxis, TestDir);
      Mat4x4 RotLS = Mat4x4::Transpose(InvRotLS);
      Vec2 Min(+FLT_MAX), Max(-FLT_MAX);
      for(int j=0; j<nVertices; ++j)
      {
        Vec2 t = pVertices[j]*RotLS;
        Min = Vec2::Min(Min, t);
        Max = Vec2::Max(Max, t);
      }
      Vec2 t = Max - Min;
      float OBBArea = t.x*t.y;
      if(BestOBBArea>OBBArea)
      {
        BestOBBArea = OBBArea;
        BBox.m_Center = 0.5f*(Min + Max)*InvRotLS;
        BBox.m_Ext[0] = 0.5f*t.x*InvRotLS.r[0];
        BBox.m_Ext[1] = 0.5f*t.y*InvRotLS.r[1];
      }
      Vec2 e = Vec2::Normalize(pFrustumHull[i%nHullSize] - pFrustumHull[(i + 1)%nHullSize]);
      TestDir = Vec3::Cross(c_ZAxis, e);
    }
    return BBox;
  }
};

#endif //#ifndef __OBB
