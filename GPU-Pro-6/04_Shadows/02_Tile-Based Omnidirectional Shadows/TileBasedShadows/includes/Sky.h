#ifndef SKY_H
#define SKY_H

#include <IPostProcessor.h>

class OGL_RenderTarget;
class RenderTargetConfig;
class OGL_Shader;
class OGL_DepthStencilState;

// Sky
//
// Extremely simple sky post-processor. Since all previously rendered opaque geometry  
// had incremented the stencil buffer, for the sky a constant colored full-screen quad 
// is only rendered where the stencil buffer is still 0.
class Sky: public IPostProcessor
{
public: 
  Sky():
    sceneRT(NULL),
    rtConfig(NULL),
    skyShader(NULL),
    depthStencilState(NULL)
  {
    strcpy(name, "Sky");
  }

  virtual bool Create() override;

  virtual OGL_RenderTarget* GetOutputRT() const override
  {
    return sceneRT;
  }

  virtual void Execute() override;

private:
  OGL_RenderTarget *sceneRT;
  RenderTargetConfig *rtConfig;
  OGL_Shader *skyShader;
  OGL_DepthStencilState *depthStencilState;

};

#endif