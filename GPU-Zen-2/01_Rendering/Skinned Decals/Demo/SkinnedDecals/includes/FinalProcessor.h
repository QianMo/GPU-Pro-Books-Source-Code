#ifndef FINAL_PROCESSOR_H
#define FINAL_PROCESSOR_H

#include <IPostProcessor.h>
#include <DX12_ResourceDescTable.h>

class DX12_RenderTarget;
class DX12_PipelineState;

// FinalProcessor
//
class FinalProcessor: public IPostProcessor
{
public: 
  FinalProcessor() :
    backBufferRT(nullptr),
    pipelineState(nullptr)
  {
    strcpy(name, "FinalProcessor");
  }

  virtual bool Create() override;

  virtual DX12_RenderTarget* GetOutputRT() const override
  {
    return backBufferRT;
  }

  virtual void Execute() override;

private:
  DX12_RenderTarget *backBufferRT;
  DX12_PipelineState *pipelineState;
  DX12_ResourceDescTable descTable;

};

#endif