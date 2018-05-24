#ifndef OGL_SAMPLER_H
#define OGL_SAMPLER_H

#include <render_states.h>

// descriptor for setting up OGL_Sampler
struct SamplerDesc
{
  SamplerDesc():
    filter(MIN_MAG_LINEAR_FILTER),
    maxAnisotropy(2),
    adressU(CLAMP_TEX_ADRESS),
    adressV(CLAMP_TEX_ADRESS),
    adressW(CLAMP_TEX_ADRESS),
    minLOD(-FLT_MAX),
    maxLOD(FLT_MAX),
    lodBias(0.0f),
    compareFunc(LEQUAL_COMP_FUNC)
  {
  }

  bool operator== (const SamplerDesc &desc) const
  {
    if(filter != desc.filter)
      return false;
    if(maxAnisotropy != desc.maxAnisotropy)
      return false;
    if(adressU != desc.adressU)
      return false;
    if(adressV != desc.adressV)
      return false;
    if(adressW != desc.adressW)
      return false;
    if(borderColor != desc.borderColor)
      return false;
    if(!IS_EQUAL(minLOD, desc.minLOD))
      return false;
    if(!IS_EQUAL(maxLOD, desc.maxLOD))
      return false;
    if(!IS_EQUAL(lodBias, desc.lodBias))
      return false;
    if(compareFunc != desc.compareFunc)
      return false;
    return true;
  }

  filterModes filter;
  unsigned int maxAnisotropy;
  texAdressModes adressU;
  texAdressModes adressV;
  texAdressModes adressW;
  Vector4 borderColor;
  float minLOD;
  float maxLOD;
  float lodBias;
  comparisonFuncs compareFunc;
};

// OGL_Sampler
//
class OGL_Sampler
{
public:
  OGL_Sampler():
    samplerName(0)
  {
  }

  ~OGL_Sampler()
  {
    Release();
  }

  void Release();

  bool Create(const SamplerDesc &desc);

  void Bind(textureBP bindingPoint) const;

  const SamplerDesc& GetDesc() const
  {
    return desc;
  }

private:
  GLuint samplerName; 
  SamplerDesc desc;
  
};

#endif