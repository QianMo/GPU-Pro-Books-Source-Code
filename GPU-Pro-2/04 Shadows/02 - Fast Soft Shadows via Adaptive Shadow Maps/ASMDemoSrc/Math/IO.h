/* Locale-independent string-to-number conversion.

   Pavlo Turchyn <pavlo@nichego-novogo.net> Aug 2010 */

#ifndef ___MATH_IO
#define ___MATH_IO

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
};

#endif //#ifndef ___MATH_IO
