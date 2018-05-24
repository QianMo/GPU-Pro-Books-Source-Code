#include <stdafx.h>
#include <Demo.h>
#include <OGL_BlendState.h>

bool OGL_BlendState::Create(const BlendDesc &desc)
{
  this->desc = desc;
  return true;
}

void OGL_BlendState::Set() const
{
  if(!desc.blend)
    glDisable(GL_BLEND); 
  else
  {
    glEnable(GL_BLEND);
    glBlendEquationSeparate(desc.blendColorOp, desc.blendAlphaOp);
    glBlendFuncSeparate(desc.srcColorBlend, desc.dstColorBlend, desc.srcAlphaBlend, desc.dstAlphaBlend);
    glBlendColor(desc.constBlendColor.x, desc.constBlendColor.y, desc.constBlendColor.z, desc.constBlendColor.w);
  }

  GLboolean red = (desc.colorMask & RED_COLOR_MASK) ? 1 : 0;
  GLboolean green = (desc.colorMask & GREEN_COLOR_MASK) ? 1 : 0;
  GLboolean blue = (desc.colorMask & BLUE_COLOR_MASK) ? 1 : 0;
  GLboolean alpha = (desc.colorMask & ALPHA_COLOR_MASK) ? 1 : 0;
  glColorMask(red, green, blue, alpha);
}


