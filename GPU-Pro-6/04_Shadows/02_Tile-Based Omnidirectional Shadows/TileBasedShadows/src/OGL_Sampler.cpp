#include <stdafx.h>
#include <Demo.h>
#include <OGL_Sampler.h>

void OGL_Sampler::Release()
{
  if(samplerName > 0)
    glDeleteSamplers(1, &samplerName);
}

bool OGL_Sampler::Create(const SamplerDesc &desc)
{
  this->desc = desc;
  glGenSamplers(1, &samplerName); 

  switch(desc.filter)
  {
  case MIN_MAG_POINT_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    break;

  case MIN_POINT_MAG_LINEAR_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    break;

  case MIN_LINEAR_MAG_POINT_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    break;

  case MIN_MAG_LINEAR_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    break;

  case MIN_MAG_MIP_POINT_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    break;

  case MIN_MAG_POINT_MIP_LINEAR_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    break;

  case MIN_POINT_MAG_LINEAR_MIP_POINT_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    break;

  case MIN_POINT_MAG_MIP_LINEAR_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    break;

  case MIN_LINEAR_MAG_MIP_POINT_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    break;

  case MIN_LINEAR_MAG_POINT_MIP_LINEAR_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    break;

  case MIN_MAG_LINEAR_MIP_POINT_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    break;

  case MIN_MAG_MIP_LINEAR_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER ,GL_LINEAR_MIPMAP_LINEAR); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    break;

  case ANISOTROPIC_FILTER:
    glSamplerParameterf(samplerName, GL_TEXTURE_MAX_ANISOTROPY_EXT, (float)desc.maxAnisotropy);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    break;

  case COMP_MIN_MAG_POINT_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    break;

  case COMP_MIN_POINT_MAG_LINEAR_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    break;

  case COMP_MIN_LINEAR_MAG_POINT_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    break;

  case COMP_MIN_MAG_LINEAR_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    break;

  case COMP_MIN_MAG_MIP_POINT_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    break;

  case COMP_MIN_MAG_POINT_MIP_LINEAR_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    break;

  case COMP_MIN_POINT_MAG_LINEAR_MIP_POINT_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    break;

  case COMP_MIN_POINT_MAG_MIP_LINEAR_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    break;

  case COMP_MIN_LINEAR_MAG_MIP_POINT_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    break;

  case COMP_MIN_LINEAR_MAG_POINT_MIP_LINEAR_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    break;

  case COMP_MIN_MAG_LINEAR_MIP_POINT_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    break;

  case COMP_MIN_MAG_MIP_LINEAR_FILTER:
    glSamplerParameteri(samplerName, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); 
    glSamplerParameteri(samplerName, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    break;

  case COMP_ANISOTROPIC_FILTER:
    glSamplerParameterf(samplerName, GL_TEXTURE_MAX_ANISOTROPY_EXT, (float)desc.maxAnisotropy);
    glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    break;
  }
  glSamplerParameteri(samplerName, GL_TEXTURE_WRAP_S, desc.adressU);
  glSamplerParameteri(samplerName, GL_TEXTURE_WRAP_T, desc.adressV);
  glSamplerParameteri(samplerName, GL_TEXTURE_WRAP_R, desc.adressW);
  glSamplerParameterfv(samplerName, GL_TEXTURE_BORDER_COLOR, desc.borderColor);
  glSamplerParameterf(samplerName, GL_TEXTURE_MIN_LOD, desc.minLOD);
  glSamplerParameterf(samplerName, GL_TEXTURE_MAX_LOD, desc.maxLOD);
  glSamplerParameterf(samplerName, GL_TEXTURE_LOD_BIAS, desc.lodBias);
  glSamplerParameteri(samplerName, GL_TEXTURE_COMPARE_FUNC, desc.compareFunc);
 
  return true;
}

void OGL_Sampler::Bind(textureBP bindingPoint) const
{
  glBindSampler(bindingPoint, samplerName);
}


