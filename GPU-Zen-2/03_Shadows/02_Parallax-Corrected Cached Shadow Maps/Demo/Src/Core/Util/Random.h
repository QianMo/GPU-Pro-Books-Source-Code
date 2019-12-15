#ifndef __RANDOM
#define __RANDOM

class RandomLCG
{
public:
  static const unsigned c_MaxUINT = 0x7fff;

  RandomLCG(float fMin=0.0f, float fMax=1.0f) : m_Number(1), m_fMin(fMin)
  {
    m_fScale = (fMax - fMin)/(float)c_MaxUINT;
  }
  void SetSeed(unsigned n)
  {
    m_Number = n;
  }
  finline unsigned GetUInt()
  {
    m_Number = m_Number*214013UL + 2531011UL;
    return (m_Number>>16)&c_MaxUINT;
  }
  finline float GetFloat()
  {
    return float(GetUInt())*m_fScale + m_fMin;
  }

private:
  unsigned m_Number;
  float m_fScale, m_fMin;
};

#endif //#ifndef __RANDOM
