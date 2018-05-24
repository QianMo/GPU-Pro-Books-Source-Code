#ifndef FINAL_PROCESSOR_H
#define FINAL_PROCESSOR_H

#include <IPostProcessor.h>

class OGL_RenderTarget;
class RenderTargetConfig;
class OGL_Shader;

// FinalProcessor
//
// Copies content of the accumulation buffer (of the GBuffers) into the back buffer 
// while performing tone-mapping.
class FinalProcessor: public IPostProcessor
{
public: 
  FinalProcessor():
    sceneRT(NULL),
    backBufferRT(NULL),
    rtConfig(NULL),
    finalPassShader(NULL)
  {
    strcpy(name, "FinalProcessor");
  }

  virtual bool Create() override;

  virtual OGL_RenderTarget* GetOutputRT() const override
  {
    return backBufferRT;
  }

  virtual void Execute() override;

private:
  OGL_RenderTarget *sceneRT;
  OGL_RenderTarget *backBufferRT;
  RenderTargetConfig *rtConfig;
  OGL_Shader *finalPassShader;

};

#endif