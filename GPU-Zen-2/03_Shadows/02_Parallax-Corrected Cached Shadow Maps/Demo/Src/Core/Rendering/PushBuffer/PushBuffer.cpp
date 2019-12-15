#include "PreCompile.h"
#include "PushBuffer.h"
#include "Platform11/Platform11.h"

template<unsigned N, void (DeviceContext11::*SetBuffer)(unsigned, ID3D11Buffer*)> class ConstantBuffers
{
public:
  ConstantBuffers(DeviceContext11* dc) : m_DC(dc)
  {
    memset(m_Buffers, 0, sizeof(m_Buffers));
  }
  ~ConstantBuffers()
  {
    for(unsigned i=0; i<N; ++i)
    {
      if(m_Buffers[i]!=NULL)
      {
        (m_DC->*SetBuffer)(i, NULL);
        m_DC->GetConstantBuffers().Free(m_Buffers[i]);
      }
    }
  }
  finline void Set(unsigned Slot, const void* pData, size_t dataSize)
  {
    _ASSERT(Slot<N);
    if(m_Buffers[Slot]!=NULL)
    {
      m_DC->GetConstantBuffers().Free(m_Buffers[Slot]);
      m_Buffers[Slot] = NULL;
    }
    if(dataSize)
      m_Buffers[Slot] = m_DC->GetConstantBuffers().Allocate(dataSize, pData, m_DC->DoNotFlushToDevice());
    (m_DC->*SetBuffer)(Slot, m_Buffers[Slot]);
  }
  finline void Set(unsigned Slot, ID3D11Buffer* pBuffer)
  {
    _ASSERT(Slot<N && pBuffer!=NULL);
    if(m_Buffers[Slot]!=NULL)
    {
      m_DC->GetConstantBuffers().Free(m_Buffers[Slot]);
      m_Buffers[Slot] = NULL;
    }
    m_Buffers[Slot] = pBuffer;
    (m_DC->*SetBuffer)(Slot, m_Buffers[Slot]);
  }

protected:
  DeviceContext11* m_DC;
  ID3D11Buffer* m_Buffers[N];
};

void PushBuffer::Execute(DeviceContext11& dc, unsigned nInstances)
{
  ConstantBuffers<DeviceContext11::PS_CONSTANT_BUFFER_SLOTS, &DeviceContext11::PSSetConstantBuffer> PSBuffers(&dc);
  ConstantBuffers<DeviceContext11::GS_CONSTANT_BUFFER_SLOTS, &DeviceContext11::GSSetConstantBuffer> GSBuffers(&dc);
  ConstantBuffers<DeviceContext11::VS_CONSTANT_BUFFER_SLOTS, &DeviceContext11::VSSetConstantBuffer> VSBuffers(&dc);
  ConstantBuffers<DeviceContext11::CS_CONSTANT_BUFFER_SLOTS, &DeviceContext11::CSSetConstantBuffer> CSBuffers(&dc);
  ConstantBuffers<DeviceContext11::HS_CONSTANT_BUFFER_SLOTS, &DeviceContext11::HSSetConstantBuffer> HSBuffers(&dc);
  ConstantBuffers<DeviceContext11::DS_CONSTANT_BUFFER_SLOTS, &DeviceContext11::DSSetConstantBuffer> DSBuffers(&dc);
  size_t offset = 0;
  for(unsigned i=0; i<m_CommandsCnt; ++i)
  {
    unsigned Cmd = m_CommandBuffer.Ptr<Command>(offset)->Cmd;
    switch(Cmd)
    {
    case IDSetContext:
      m_CommandBuffer.Ptr<SetContext>(offset)->Context.ApplyTo(dc);
      break;
    case IDDraw:          Issue(dc, m_CommandBuffer.Ptr<CommandDraw>(offset)); break;
    case IDDrawIndexed:   Issue(dc, m_CommandBuffer.Ptr<CommandDrawIndexed>(offset)); break;
    case IDDrawInstanced: Issue(dc, m_CommandBuffer.Ptr<CommandDrawInstanced>(offset)); break;
    case IDDrawIndexedInstanced: Issue(dc, m_CommandBuffer.Ptr<CommandDrawIndexedInstanced>(offset)); break;
    case IDDrawIndexedInstancedVariant: Issue(dc, m_CommandBuffer.Ptr<CommandDrawIndexedInstancedVariant>(offset), nInstances); break;
    case IDDispatch:      Issue(dc, m_CommandBuffer.Ptr<CommandDispatch>(offset)); break;
    case IDSetConstantPS: Issue(PSBuffers, m_CommandBuffer.Ptr<SetConstant>(offset)); break;
    case IDSetConstantGS: Issue(GSBuffers, m_CommandBuffer.Ptr<SetConstant>(offset)); break;
    case IDSetConstantVS: Issue(VSBuffers, m_CommandBuffer.Ptr<SetConstant>(offset)); break;
    case IDSetConstantCS: Issue(CSBuffers, m_CommandBuffer.Ptr<SetConstant>(offset)); break;
    case IDSetConstantHS: Issue(HSBuffers, m_CommandBuffer.Ptr<SetConstant>(offset)); break;
    case IDSetConstantDS: Issue(DSBuffers, m_CommandBuffer.Ptr<SetConstant>(offset)); break;
    default: _ASSERT(!"corrupt command buffer"); return;
    }
    offset += GET_CMD_SIZE(Cmd);
  }
}

void PushBuffer::Append(const PushBuffer& a, unsigned nUpdateHandles, PushBufferConstantHandle* pUpdateHandles)
{
  Barrier();

  size_t deltaData = m_DataBuffer.Size();
  m_DataBuffer.Write(a.m_DataBuffer.Ptr<void>(0), a.m_DataBuffer.Size());

  size_t doneSize = (nUpdateHandles + sizeof(unsigned)*8 - 1)/(sizeof(unsigned)*8);
  unsigned* done = (unsigned*)alloca(doneSize*sizeof(unsigned));
  memset(done, 0, doneSize*sizeof(unsigned));
  size_t offset = 0;
  for(unsigned i=0; i<a.m_CommandsCnt; ++i)
  {
    unsigned Cmd = a.m_CommandBuffer.Ptr<Command>(offset)->Cmd;
    switch(Cmd)
    {
    case IDSetContext:    PushRC(a.m_CommandBuffer.Ptr<SetContext>(offset)->Context); break;
    case IDDraw:          PushCommand(*a.m_CommandBuffer.Ptr<CommandDraw>(offset)); break;
    case IDDrawIndexed:   PushCommand(*a.m_CommandBuffer.Ptr<CommandDrawIndexed>(offset)); break;
    case IDDrawInstanced: PushCommand(*a.m_CommandBuffer.Ptr<CommandDrawInstanced>(offset)); break;
    case IDDrawIndexedInstanced: PushCommand(*a.m_CommandBuffer.Ptr<CommandDrawIndexedInstanced>(offset)); break;
    case IDDrawIndexedInstancedVariant: PushCommand(*a.m_CommandBuffer.Ptr<CommandDrawIndexedInstancedVariant>(offset)); break;
    case IDDispatch:      PushCommand(*a.m_CommandBuffer.Ptr<CommandDispatch>(offset)); break;
    case IDSetConstantPS:
    case IDSetConstantGS:
    case IDSetConstantVS:
    case IDSetConstantCS:
    case IDSetConstantHS:
    case IDSetConstantDS:
      {
        SetConstant c = *a.m_CommandBuffer.Ptr<SetConstant>(offset);
        c.DataOffset += deltaData;
        size_t commandOffset = PushCommand(c);
        if(c.pBuffer!=NULL)
          c.pBuffer->AddRef();
        for(unsigned i=0; i<nUpdateHandles; ++i)
        {
          if(!(done[i>>5] & (1<<(i&31))) && pUpdateHandles[i].m_CommandOffset==offset)
          {
            pUpdateHandles[i].m_DataOffset += deltaData;
            pUpdateHandles[i].m_CommandOffset = commandOffset;
            done[i>>5] |= (1<<(i&31));
          }
        }
      }
      break;
    default: _ASSERT(!"corrupt command buffer"); return;
    }
    offset += GET_CMD_SIZE(Cmd);
  }
  for(unsigned i=0; i<nUpdateHandles; ++i)
  {
    _ASSERT(done[i>>5] & (1<<(i&31)));
    _ASSERT(pUpdateHandles[i].IsValid(&a, 0, a.m_ClearID));
#ifdef PARANOID_PUSH_BUFFER
    pUpdateHandles[i].m_pOwner = this;
    pUpdateHandles[i].m_ClearID = m_ClearID;
#endif
  }
}

void PushBuffer::Clear()
{
  DeleteConstantBuffers();
  m_CommandBuffer.Resize(0);
  m_DataBuffer.Resize(0);
  m_CommandsCnt = 0;
  m_LastSetContextCommand = -1;
  Barrier();
  static int s_IDGen = 0;
  m_ClearID = s_IDGen++;
}

void PushBuffer::CreateConstantBuffers()
{
  size_t offset = 0;
  for(unsigned i=0; i<m_CommandsCnt; ++i)
  {
    unsigned Cmd = m_CommandBuffer.Ptr<Command>(offset)->Cmd;
    switch(Cmd)
    {
    case IDSetConstantPS:
    case IDSetConstantGS:
    case IDSetConstantVS:
    case IDSetConstantCS:
    case IDSetConstantHS:
    case IDSetConstantDS:
      if(m_CommandBuffer.Ptr<SetConstant>(offset)->pBuffer==NULL)
      {
        SetConstant* c = m_CommandBuffer.Ptr<SetConstant>(offset);
        Buffer11 buf;
        HRESULT hr = buf.Init(Platform::GetD3DDevice(), c->DataSize, 1, m_DataBuffer.Ptr<void>(c->DataOffset));
        _ASSERT(SUCCEEDED(hr));
        c->pBuffer = buf.GetBuffer();
      }
      break;
    }
    offset += GET_CMD_SIZE(Cmd);
  }
}

void PushBuffer::DeleteConstantBuffers()
{
  size_t offset = 0;
  for(unsigned i=0; i<m_CommandsCnt; ++i)
  {
    unsigned Cmd = m_CommandBuffer.Ptr<Command>(offset)->Cmd;
    switch(Cmd)
    {
    case IDSetConstantPS:
    case IDSetConstantGS:
    case IDSetConstantVS:
    case IDSetConstantCS:
    case IDSetConstantHS:
    case IDSetConstantDS:
      SAFE_RELEASE(m_CommandBuffer.Ptr<SetConstant>(offset)->pBuffer);
      break;
    }
    offset += GET_CMD_SIZE(Cmd);
  }
}
