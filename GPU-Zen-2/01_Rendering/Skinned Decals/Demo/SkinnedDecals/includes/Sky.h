#ifndef SKY_H
#define SKY_H

#include <IPostProcessor.h>

class DX12_PipelineState;

// Sky
//
class Sky: public IPostProcessor
{
public: 
  Sky():
    pipelineState(nullptr)
  {
    strcpy(name, "Sky");
  }

  virtual bool Create() override;

  virtual DX12_RenderTarget* GetOutputRT() const override
  {
    return nullptr;
  }

  virtual void Execute() override;

private:
  DX12_PipelineState *pipelineState;

};

#endif