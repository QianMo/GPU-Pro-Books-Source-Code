#ifndef __PREALLOCATED_PUSH_BUFFER_H
#define __PREALLOCATED_PUSH_BUFFER_H

#include "PushBuffer.h"

#pragma warning(push)
#pragma warning(disable:4324)

template<unsigned c_CommandBufferSize=4096, unsigned c_DataBufferSize=2048> 
  class PreallocatedPushBuffer : public PushBuffer, public MathLibObject
{
public:
  PreallocatedPushBuffer() : 
    PushBuffer(c_CommandBufferSize, m_PreallocatedCommandBuffer, c_DataBufferSize, m_PreallocatedDataBuffer) { }

protected:
  align16 unsigned char m_PreallocatedCommandBuffer[c_CommandBufferSize];
  align16 unsigned char m_PreallocatedDataBuffer[c_DataBufferSize];
};

#pragma warning(pop)

#endif //#ifndef __PREALLOCATED_PUSH_BUFFER_H
