#ifndef IPOST_PROCESSOR_H
#define IPOST_PROCESSOR_H

class DX12_RenderTarget;

// IPostProcessor
//
// Interface for post-processors.
class IPostProcessor
{
public:
  IPostProcessor():
    active(true)
  {
    name[0] = 0;
  }

  virtual ~IPostProcessor()
  {
  }

  virtual bool Create()=0;

  virtual DX12_RenderTarget* GetOutputRT() const=0;

  virtual void Execute()=0;

  const char* GetName() const
  {
    return name;
  }

  void SetActive(bool active)
  {
    this->active = active;
  }

  bool IsActive() const
  {
    return active;
  }

protected:
  char name[DEMO_MAX_STRING];
  bool active; 

};

#endif
