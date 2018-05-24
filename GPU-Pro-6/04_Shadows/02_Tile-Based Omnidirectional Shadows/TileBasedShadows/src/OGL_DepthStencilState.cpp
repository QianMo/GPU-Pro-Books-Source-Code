#include <stdafx.h>
#include <Demo.h>
#include <OGL_DepthStencilState.h>

bool OGL_DepthStencilState::Create(const DepthStencilDesc &desc)
{
  this->desc = desc;
  return true;
}

void OGL_DepthStencilState::Set() const
{
  if(desc.depthTest)
  {
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(desc.depthFunc);
  }
  else
    glDisable(GL_DEPTH_TEST);

  glDepthMask(desc.depthMask);

  if(!desc.stencilTest)
    glDisable(GL_STENCIL_TEST);
  else
  {
    glEnable(GL_STENCIL_TEST);	
    glStencilFunc(desc.stencilFunc, desc.stencilRef, desc.stencilMask);
    glStencilOp(desc.stencilFailOp, desc.stencilDepthFailOp, desc.stencilPassOp);
  }
}

