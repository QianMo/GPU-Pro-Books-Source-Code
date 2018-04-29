#ifndef __CAMERA_PATH
#define __CAMERA_PATH

#include "Math/Math.h"
#include "Util/AlignedVector.h"

class CameraPath
{
public:
  CameraPath(const char* pszFileName)
  {
    m_Keys.reserve(128);
    FILE* pFile;
    if(!fopen_s(&pFile, pszFileName, "r"))
    {
      while(!feof(pFile))
      {
        char pBuf[4096];
        if(fgets(pBuf, sizeof(pBuf), pFile))
        {
          char* pLine = pBuf;
          char* pTokens[256];
          char* pContext;
          int nTokens = 0;
          while(pTokens[nTokens] = strtok_s(pLine, " \n", &pContext))
          {
            if(++nTokens>=ARRAYSIZE(pTokens)) break;
            pLine = NULL;
          }
          if(nTokens!=7) { m_Keys.clear(); break; }
          float m[7];
          for(int i=0; i<7; ++i)
            m[i] = MathIO::ReadFloat(pTokens[i]);
          Key k;
          k.r = Quat(c_YAxis)*Quat::Conjugate(Quat(&m[0]));
          k.t = -Vec3(&m[4]);
          m_Keys.push_back(k);
        }
      }
      fclose(pFile);
      m_TrackLengthMS = m_Keys.size()*1000/KEYS_PER_SECOND;
    }
  }
  Mat4x4 GetTransform(unsigned nTimeMS) const
  {
    nTimeMS = nTimeMS%m_TrackLengthMS;
    unsigned i = nTimeMS*KEYS_PER_SECOND/1000;
    float f = (float)(nTimeMS - i*1000/KEYS_PER_SECOND)/(float)(1000/KEYS_PER_SECOND);
    const Key& Key0 = m_Keys[i];
    const Key& Key1 = m_Keys[(i + 1)%m_Keys.size()];
    D3DXQUATERNION r, r0(&Key0.r.x), r1(&Key1.r.x);
    D3DXQuaternionSlerp(&r, &r0, &r1, f);
    return Mat4x4::TranslationD3D((1.0f - f)*Key0.t + f*Key1.t)*Quat::AsMatrixD3D(Quat(&r.x));
  }

private:
  static const int KEYS_PER_SECOND = 4;
  struct Key { Quat r; Vec3 t; };

  AlignedPODVector<Key> m_Keys;
  unsigned m_TrackLengthMS;
};

#endif //#ifndef __CAMERA_PATH
