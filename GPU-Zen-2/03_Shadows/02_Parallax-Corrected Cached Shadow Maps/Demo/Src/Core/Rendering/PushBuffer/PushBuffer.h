#ifndef __PUSH_BUFFER_H
#define __PUSH_BUFFER_H

#include "Platform11/Platform11.h"
#include "../../Util/MemoryBuffer.h"

#ifndef RUNTIME_CHECKS_OFF
  #define PARANOID_PUSH_BUFFER
#endif

class PushBuffer;

class PushBufferConstantHandle
{
public:
  finline PushBufferConstantHandle() : m_DataOffset(c_InvalidOffset), m_CommandOffset(c_InvalidOffset) { }
  finline bool IsValid() const { return (m_DataOffset!=c_InvalidOffset) & (m_CommandOffset!=c_InvalidOffset); }

protected:
  static const unsigned c_InvalidOffset = ~0U;

  unsigned m_DataOffset;
  unsigned m_CommandOffset;
#ifdef PARANOID_PUSH_BUFFER
  PushBuffer* m_pOwner;
  unsigned m_Size;
  unsigned m_ClearID;

  finline PushBufferConstantHandle(unsigned dataOffset, unsigned commandOffset, PushBuffer* p, unsigned size, unsigned clearID) : 
    m_DataOffset(dataOffset), m_CommandOffset(commandOffset), m_pOwner(p), m_Size(size), m_ClearID(clearID) { }
  finline bool IsValid(const PushBuffer* p, unsigned size, unsigned clearID) const { return IsValid() & (m_pOwner==p) & (size<=m_Size) & (m_ClearID==clearID); }
#else
  finline PushBufferConstantHandle(unsigned dataOffset, unsigned commandOffset, PushBuffer*, unsigned, unsigned) :
    m_DataOffset(dataOffset), m_CommandOffset(commandOffset) { }
  finline bool IsValid(const PushBuffer*, unsigned, unsigned) const { return IsValid(); }
#endif

  friend class PushBuffer;
};

#define MAKE_PB_CMD(id, size) (((id)<<16) + (size))
#define GET_CMD_SIZE(cmd)     ((cmd)&0xffff)

class PushBuffer
{
public:
  PushBuffer(size_t commandBufferSize, void* pCommandBuffer, size_t dataBufferSize, void* pDataBuffer) :
    m_CommandBuffer(commandBufferSize, pCommandBuffer), m_DataBuffer(dataBufferSize, pDataBuffer), m_CommandsCnt(0)
  {
    Clear();
  }
  PushBuffer() : m_CommandsCnt(0)
  {
    Clear();
  }
  ~PushBuffer()
  {
    DeleteConstantBuffers();
  }

  finline bool IsEmpty() const { return m_CommandsCnt==0; }

  template<class T> finline void PushConstantPS(const T& data, int Slot = -1) { PushConstant<IDSetConstantPS>(data, Slot); }
  template<class T> finline void PushConstantGS(const T& data, int Slot = -1) { PushConstant<IDSetConstantGS>(data, Slot); }
  template<class T> finline void PushConstantVS(const T& data, int Slot = -1) { PushConstant<IDSetConstantVS>(data, Slot); }
  template<class T> finline void PushConstantCS(const T& data, int Slot = -1) { PushConstant<IDSetConstantCS>(data, Slot); }
  template<class T> finline void PushConstantHS(const T& data, int Slot = -1) { PushConstant<IDSetConstantHS>(data, Slot); }
  template<class T> finline void PushConstantDS(const T& data, int Slot = -1) { PushConstant<IDSetConstantDS>(data, Slot); }

  finline PushBufferConstantHandle PushConstantPS(const void* pData, size_t dataSize, int Slot = -1) { return PushConstant<IDSetConstantPS>(dataSize, pData, Slot); }
  finline PushBufferConstantHandle PushConstantGS(const void* pData, size_t dataSize, int Slot = -1) { return PushConstant<IDSetConstantGS>(dataSize, pData, Slot); }
  finline PushBufferConstantHandle PushConstantVS(const void* pData, size_t dataSize, int Slot = -1) { return PushConstant<IDSetConstantVS>(dataSize, pData, Slot); }
  finline PushBufferConstantHandle PushConstantCS(const void* pData, size_t dataSize, int Slot = -1) { return PushConstant<IDSetConstantCS>(dataSize, pData, Slot); }
  finline PushBufferConstantHandle PushConstantHS(const void* pData, size_t dataSize, int Slot = -1) { return PushConstant<IDSetConstantHS>(dataSize, pData, Slot); }
  finline PushBufferConstantHandle PushConstantDS(const void* pData, size_t dataSize, int Slot = -1) { return PushConstant<IDSetConstantDS>(dataSize, pData, Slot); }

  finline void Update(const PushBufferConstantHandle& h, size_t dataSize, const void* pData, bool updateDataSize = false)
  {
    _ASSERT(h.IsValid(this, dataSize, m_ClearID));
    m_DataBuffer.Seek(h.m_DataOffset);
    m_DataBuffer.Write(pData, dataSize);
    m_DataBuffer.Seek(m_DataBuffer.Size());
    SetConstant* c = m_CommandBuffer.Ptr<SetConstant>(h.m_CommandOffset);
    SAFE_RELEASE(c->pBuffer);
    if(updateDataSize)
      c->DataSize = h.m_DataOffset - c->DataOffset + dataSize;
  }
  template<class T> finline void Update(const PushBufferConstantHandle& h, const T& data, bool updateDataSize = false)
  {
    _ASSERT(h.IsValid(this, sizeof(data), m_ClearID));
    m_DataBuffer.Seek(h.m_DataOffset);
    m_DataBuffer.Write(data);
    m_DataBuffer.Seek(m_DataBuffer.Size());
    SetConstant* c = m_CommandBuffer.Ptr<SetConstant>(h.m_CommandOffset);
    SAFE_RELEASE(c->pBuffer);
    if(updateDataSize)
      c->DataSize = h.m_DataOffset - c->DataOffset + sizeof(T);
  }
  finline void PushRC(const RenderContext11& rc)
  {
    if(m_LastSetContextCommand<0 || !(m_CommandBuffer.Ptr<SetContext>(m_LastSetContextCommand)->Context==rc))
    {
      SetContext c; c.Cmd = IDSetContext;
      c.Context = rc;
      m_LastSetContextCommand = PushCommand(c);
    }
  }
  finline void Draw(unsigned VertexCount, unsigned StartVertexLocation)
  {
    CommandDraw c; c.Cmd = IDDraw;
    c.VertexCount = VertexCount; c.StartVertexLocation = StartVertexLocation;
    PushCommand(c); Barrier();
  }
  finline void DrawIndexed(unsigned IndexCount, unsigned StartIndexLocation, int BaseVertexLocation)
  {
    CommandDrawIndexed c; c.Cmd = IDDrawIndexed;
    c.IndexCount = IndexCount; c.StartIndexLocation = StartIndexLocation; c.BaseVertexLocation = BaseVertexLocation;
    PushCommand(c); Barrier();
  }
  finline void DrawInstanced(unsigned VertexCountPerInstance, unsigned InstanceCount, unsigned StartVertexLocation, unsigned StartInstanceLocation)
  {
    CommandDrawInstanced c; c.Cmd = IDDrawInstanced;
    c.VertexCountPerInstance = VertexCountPerInstance; c.InstanceCount = InstanceCount; c.StartVertexLocation = StartVertexLocation; c.StartInstanceLocation = StartInstanceLocation;
    PushCommand(c); Barrier();
  }
  finline void DrawIndexedInstanced(unsigned IndexCountPerInstance, unsigned InstanceCount, unsigned StartIndexLocation, int BaseVertexLocation, unsigned StartInstanceLocation)
  {
    CommandDrawIndexedInstanced c; c.Cmd = IDDrawIndexedInstanced;
    c.IndexCountPerInstance = IndexCountPerInstance; c.InstanceCount = InstanceCount; c.StartIndexLocation = StartIndexLocation; c.BaseVertexLocation = BaseVertexLocation; c.StartInstanceLocation = StartInstanceLocation;
    PushCommand(c); Barrier();
  }
  finline void DrawIndexedInstancedVariant(unsigned IndexCountPerInstance, unsigned StartIndexLocation, int BaseVertexLocation, unsigned StartInstanceLocation)
  {
    CommandDrawIndexedInstancedVariant c; c.Cmd = IDDrawIndexedInstancedVariant;
    c.IndexCountPerInstance = IndexCountPerInstance; c.StartIndexLocation = StartIndexLocation; c.BaseVertexLocation = BaseVertexLocation; c.StartInstanceLocation = StartInstanceLocation;
    PushCommand(c); Barrier();
  }
  finline void Dispatch(unsigned ThreadGroupCountX, unsigned ThreadGroupCountY, unsigned ThreadGroupCountZ)
  {
    CommandDispatch c; c.Cmd = IDDispatch;
    c.ThreadGroupCountX = ThreadGroupCountX; c.ThreadGroupCountY = ThreadGroupCountY; c.ThreadGroupCountZ = ThreadGroupCountZ;
    PushCommand(c); Barrier();
  }

  void Execute(DeviceContext11& dc = Platform::GetImmediateContext(), unsigned nInstances = 1);
  void Append(const PushBuffer&, unsigned nUpdateHandles = 0, PushBufferConstantHandle* pUpdateHandles = NULL);
  void Clear();
  void CreateConstantBuffers();
  void DeleteConstantBuffers();

protected:
  struct Command
  {
    unsigned Cmd;
  };
  struct SetContext : public Command
  {
    RenderContext11 Context;
  };
  struct SetConstant : public Command
  {
    int Slot;
    unsigned DataOffset;
    unsigned DataSize;
    ID3D11Buffer* pBuffer;
  };
  struct CommandDraw : public Command
  {
    unsigned VertexCount;
    unsigned StartVertexLocation;
  };
  struct CommandDrawIndexed : public Command
  {
    unsigned IndexCount;
    unsigned StartIndexLocation;
    int BaseVertexLocation;
  };
  struct CommandDrawInstanced : public Command
  {
    unsigned VertexCountPerInstance;
    unsigned InstanceCount;
    unsigned StartVertexLocation;
    unsigned StartInstanceLocation;
  };
  struct CommandDrawIndexedInstancedVariant : public Command
  {
    unsigned IndexCountPerInstance;
    unsigned StartIndexLocation;
    int BaseVertexLocation;
    unsigned StartInstanceLocation;
  };
  struct CommandDrawIndexedInstanced : public CommandDrawIndexedInstancedVariant
  {
    unsigned InstanceCount;
  };
  struct CommandDispatch : public Command
  {
    unsigned ThreadGroupCountX;
    unsigned ThreadGroupCountY;
    unsigned ThreadGroupCountZ;
  };
  enum CommandID
  {
    IDSetConstantPS = MAKE_PB_CMD(0, sizeof(SetConstant)),
    IDSetConstantGS = MAKE_PB_CMD(1, sizeof(SetConstant)),
    IDSetConstantVS = MAKE_PB_CMD(2, sizeof(SetConstant)),
    IDSetConstantCS = MAKE_PB_CMD(3, sizeof(SetConstant)),
    IDSetConstantHS = MAKE_PB_CMD(4, sizeof(SetConstant)),
    IDSetConstantDS = MAKE_PB_CMD(5, sizeof(SetConstant)),
    IDSetContext    = MAKE_PB_CMD(6, sizeof(SetContext)),
    IDDraw          = MAKE_PB_CMD(7, sizeof(CommandDraw)),
    IDDrawIndexed   = MAKE_PB_CMD(8, sizeof(CommandDrawIndexed)),
    IDDrawInstanced = MAKE_PB_CMD(9, sizeof(CommandDrawInstanced)),
    IDDrawIndexedInstanced = MAKE_PB_CMD(10, sizeof(CommandDrawIndexedInstanced)),
    IDDrawIndexedInstancedVariant = MAKE_PB_CMD(11, sizeof(CommandDrawIndexedInstancedVariant)),
    IDDispatch      = MAKE_PB_CMD(12, sizeof(CommandDispatch)),
  };

  MemoryBuffer m_CommandBuffer;
  MemoryBuffer m_DataBuffer;
  unsigned m_CommandsCnt;

  int m_LastSetConstantCommand;
  int m_LastSetContextCommand;
  unsigned m_ClearID;

  template<class T> finline unsigned PushCommand(const T& cmd)
  {
    size_t offset = m_CommandBuffer.Position();
    m_CommandBuffer.Write(cmd);
    ++m_CommandsCnt;
    return offset;
  }
  template<class T> finline unsigned PushData(const T& d)
  {
    size_t offset = m_DataBuffer.Position();
    m_DataBuffer.Write(d);
    return offset;
  }
  finline unsigned PushData(const void* pData, unsigned dataSize)
  {
    size_t offset = m_DataBuffer.Position();
    if(pData!=NULL)
    {
      m_DataBuffer.Write(pData, dataSize);
    }
    else
    {
      m_DataBuffer.Resize(m_DataBuffer.Size() + dataSize);
      m_DataBuffer.Seek(m_DataBuffer.Size());
    }
    return offset;
  }
  template<CommandID CMD, class T> finline void PushConstant(const T& d, int Slot)
  {
    if(Slot<0)
    {
      if(m_LastSetConstantCommand>=0)
      {
        SetConstant& c = *m_CommandBuffer.Ptr<SetConstant>(m_LastSetConstantCommand);
        if((c.Cmd==CMD) & ((c.DataOffset + c.DataSize)==m_DataBuffer.Size()))
        {
          PushData(d);
          c.DataSize += sizeof(d);
          SAFE_RELEASE(c.pBuffer);
          return;
        }
      }
      _ASSERT(!"slot is undefined");
      return;
    }
    SetConstant c;
    c.Cmd = CMD;
    c.Slot = Slot;
    c.DataOffset = PushData(d);
    c.DataSize = sizeof(d);
    c.pBuffer = NULL;
    m_LastSetConstantCommand = PushCommand(c);
  }
  template<CommandID CMD> finline PushBufferConstantHandle PushConstant(unsigned dataSize, const void* pData, int Slot)
  {
    if(Slot<0)
    {
      if(m_LastSetConstantCommand>=0)
      {
        SetConstant& c = *m_CommandBuffer.Ptr<SetConstant>(m_LastSetConstantCommand);
        if((c.Cmd==CMD) & ((c.DataOffset + c.DataSize)==m_DataBuffer.Size()))
        {
          c.DataSize += dataSize;
          SAFE_RELEASE(c.pBuffer);
          return PushBufferConstantHandle(PushData(pData, dataSize), m_LastSetConstantCommand, this, dataSize, m_ClearID);
        }
      }
      _ASSERT(!"slot is undefined");
      return PushBufferConstantHandle();
    }
    SetConstant c;
    c.Cmd = CMD;
    c.Slot = Slot;
    c.DataOffset = PushData(pData, dataSize);
    c.DataSize = dataSize;
    c.pBuffer = NULL;
    m_LastSetConstantCommand = PushCommand(c);
    return PushBufferConstantHandle(c.DataOffset, m_LastSetConstantCommand, this, dataSize, m_ClearID);
  }
  finline void Barrier()
  {
    m_LastSetConstantCommand = -1;
  }

  template<class T> static finline void Issue(DeviceContext11&, const T*) { static_assert(false, "not implemented"); }
  template<> static finline void Issue<>(DeviceContext11& dc, const CommandDraw* c)                 { dc.FlushToDevice()->Draw(c->VertexCount, c->StartVertexLocation); }
  template<> static finline void Issue<>(DeviceContext11& dc, const CommandDrawIndexed* c)          { dc.FlushToDevice()->DrawIndexed(c->IndexCount, c->StartIndexLocation, c->BaseVertexLocation); }
  template<> static finline void Issue<>(DeviceContext11& dc, const CommandDrawInstanced* c)        { dc.FlushToDevice()->DrawInstanced(c->VertexCountPerInstance, c->InstanceCount, c->StartVertexLocation, c->StartInstanceLocation); }
  template<> static finline void Issue<>(DeviceContext11& dc, const CommandDrawIndexedInstanced* c) { dc.FlushToDevice()->DrawIndexedInstanced(c->IndexCountPerInstance, c->InstanceCount, c->StartIndexLocation, c->BaseVertexLocation, c->StartInstanceLocation); }
  template<> static finline void Issue<>(DeviceContext11& dc, const CommandDispatch* c)             { dc.FlushToDevice()->Dispatch(c->ThreadGroupCountX, c->ThreadGroupCountY, c->ThreadGroupCountZ); }

  static finline void Issue(DeviceContext11& dc, const CommandDrawIndexedInstancedVariant* c, unsigned n)
  {
    if(n==1) dc.FlushToDevice()->DrawIndexed(c->IndexCountPerInstance, c->StartIndexLocation, c->BaseVertexLocation);
    else if(n>0) dc.FlushToDevice()->DrawIndexedInstanced(c->IndexCountPerInstance, n, c->StartIndexLocation, c->BaseVertexLocation, c->StartInstanceLocation);
  }
  template<class T> finline void Issue(T& cb, const SetConstant* c)
  {
    if(c->pBuffer!=NULL) cb.Set(c->Slot, c->pBuffer);
    else cb.Set(c->Slot, m_DataBuffer.Ptr<void>(c->DataOffset), c->DataSize);
  }
};

#endif //#ifndef __PUSH_BUFFER_H
