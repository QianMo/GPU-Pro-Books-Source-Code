#ifndef __DEFERRED_CONTEXT11
#define __DEFERRED_CONTEXT11

#include "DeviceContext11.h"

class DeferredContext11 : public DeviceContext11
{
public:
  DeferredContext11();
  ~DeferredContext11();
  void FinishCommandList(bool RestoreDeferredContextState = true);
  void ExecuteCommandList();

protected:
  ID3D11CommandList* m_CommandList;
};

#endif //#ifndef __DEFERRED_CONTEXT11
