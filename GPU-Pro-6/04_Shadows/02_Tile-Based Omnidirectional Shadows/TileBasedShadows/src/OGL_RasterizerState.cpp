#include <stdafx.h>
#include <Demo.h>
#include <OGL_RasterizerState.h>

#define MAX_NUM_CLIP_PLANES 6

bool OGL_RasterizerState::Create(const RasterizerDesc &desc)
{
  this->desc = desc;
  if(desc.numClipPlanes > MAX_NUM_CLIP_PLANES)
    return false;
  return true;
}

void OGL_RasterizerState::Set() const
{
  glPolygonMode(GL_FRONT_AND_BACK, desc.fillMode);

  if(desc.cullMode != NONE_CULL)
  {
    glEnable(GL_CULL_FACE);
    glCullFace(desc.cullMode);
  }
  else
    glDisable(GL_CULL_FACE);

  if(!desc.scissorTest)
    glDisable(GL_SCISSOR_TEST);
  else
    glEnable(GL_SCISSOR_TEST);

  if(!desc.multisampleEnable)
    glDisable(GL_MULTISAMPLE);
  else
    glEnable(GL_MULTISAMPLE);

  if((!IS_EQUAL(desc.depthBias, 0.0f)) || (!IS_EQUAL(desc.slopeScaledDepthBias, 0.0f)))
  {
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(desc.slopeScaledDepthBias, desc.depthBias);
  }
  else
    glDisable(GL_POLYGON_OFFSET_FILL);

  for(unsigned int i=0; i<MAX_NUM_CLIP_PLANES; i++)
  {
    if(i < desc.numClipPlanes)
      glEnable(GL_CLIP_PLANE0+i);
    else
      glDisable(GL_CLIP_PLANE0+i);
  }
}


