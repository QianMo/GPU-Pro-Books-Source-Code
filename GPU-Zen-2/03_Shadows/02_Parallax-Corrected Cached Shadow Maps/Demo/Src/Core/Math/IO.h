#ifndef ___MATH_IO
#define ___MATH_IO

#include <string.h>
#include <crtdbg.h>

class MathIO
{
public:
  static int ReadInteger(const char *p)
  {
    int s = p[0]=='-' ? -1 : 1;
    p += (p[0]=='-') | (p[0]=='+');
    int r = 0;
    for(; p[0]>='0' && p[0]<='9'; ++p)
      r = r*10 + (int)(p[0] - '0');
    return r*s;
  }
  static double ReadDouble(const char* p)
  {
    float s = p[0]=='-' ? -1.0f : 1.0f;
    p += (p[0]=='-') | (p[0]=='+');
    double r = 0;
    for(; p[0]>='0' && p[0]<='9'; ++p)
      r = r*10.0 + (double)(p[0] - '0');
    if(p[0]=='.' || p[0]==',')
    {
      double k = 0.1;
      for(++p; p[0]>='0' && p[0]<='9'; ++p, k *= 0.1)
        r += k*(double)(p[0] - '0');
    }
    if(p[0]=='e' || p[0]=='E')
      r *= pow(10.0, (double)ReadInteger(p + 1));
    return r*s;
  }
  static float ReadFloat(const char* p)
  {
    return (float)ReadDouble(p);
  }
  template<class T, T (*Read)(const char*)> static size_t ReadTabulatedArray(const char* p, T* pArray, size_t arraySize, const char* const pszSeparators = " ")
  {
    for(size_t n = 0; n<arraySize; ++n)
    {
      while(strchr(pszSeparators, *p)!=NULL)
        if(!*p++) return n;
      pArray[n] = Read(p);
      while(strchr(pszSeparators, *p)==NULL)
        if(!*p++) return (n + 1);
    }
    return arraySize;
  }
  template<class T, unsigned N> static const T Read(const char* p, const char* const pszSeparators = " ")
  {
    float f[N];
#ifndef NDEBUG
    size_t actualElements = 
#endif
    ReadTabulatedArray<float, &ReadFloat>(p, f, N, pszSeparators);
#ifndef NDEBUG
    _ASSERT(actualElements==N);
#endif
    return T(f);
  }
  template<class T, unsigned N> static const T ReadI(const char* p, const char* const pszSeparators = " ")
  {
    int i[N];
#ifndef NDEBUG
    size_t actualElements = 
#endif
    ReadTabulatedArray<int, &ReadInteger>(p, i, N, pszSeparators);
#ifndef NDEBUG
    _ASSERT(actualElements == N);
#endif
    return T(i);
  }
};

#endif //#ifndef ___MATH_IO
