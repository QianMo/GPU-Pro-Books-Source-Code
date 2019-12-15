#ifndef DX12_BUFFER_H
#define DX12_BUFFER_H

#include <DX12_IResource.h>
#include <render_states.h>

enum bufferTypes
{
  NONE_BUFFER=0,
  VERTEX_BUFFER,
  INDEX_BUFFER,
  STRUCTURED_BUFFER,
  CONSTANT_BUFFER
};

BITFLAGS_ENUM(UINT, bufferFlags)
{
  NONE_BUFFER_FLAG      = 0,
  CPU_WRITE_BUFFER_FLAG = 1,
  CPU_READ_BUFFER_FLAG  = 2,
  DYNAMIC_BUFFER_FLAG   = 4
};

struct BufferDesc
{
  BufferDesc() :
    bufferType(NONE_BUFFER),
    elementSize(0),
    numElements(0),
    elementFormat(NONE_RENDER_FORMAT),
    flags(NONE_BUFFER_FLAG)
  {
  }

  bufferTypes bufferType;
  UINT elementSize;
  UINT numElements;
  renderFormats elementFormat;
  bufferFlags flags;
};


// DX12_Buffer
//
class DX12_Buffer : public DX12_IResource
{
public:
  friend class DX12_ResourceDescTable;

  DX12_Buffer() :
    resourceState(COMMON_RESOURCE_STATE),
    mappedBuffer(nullptr),
    bufferSize(0)
  {
  }

  ~DX12_Buffer()
  {
    Release();
  }

  void Release();

  bool Create(const BufferDesc &desc, const char *name);

  bool Update(const void *bufferData, UINT elementCount);

  virtual ID3D12Resource* GetResource() const override
  {
    return buffer.Get();
  }

  virtual ID3D12Resource* GetUploadHeap() const override
  {
    return ((bufferDesc.flags & CPU_WRITE_BUFFER_FLAG) && (!(bufferDesc.flags & DYNAMIC_BUFFER_FLAG))) ? uploadHeap.Get() : nullptr;
  }

  virtual void SetResourceState(resourceStates resourceState) override
  {
    this->resourceState = resourceState;
  }

  virtual resourceStates GetResourceState() const override
  {
    return resourceState;
  }

  virtual UINT GetNumSubresources() const override
  {
    return 1;
  }

  GpuVirtualAddress GetGpuVirtualAddress() const;

  D3D12_VERTEX_BUFFER_VIEW GetVertexBufferView() const;

  D3D12_INDEX_BUFFER_VIEW GetIndexBufferView() const;

  const BufferDesc& GetBufferDesc() const
  {
    return bufferDesc;
  } 

private:
  DescHandle CreateSrv(UINT backBufferIndex) const;

  DescHandle CreateUav(UINT backBufferIndex) const;

  ComPtr<ID3D12Resource> buffer;
  ComPtr<ID3D12Resource> uploadHeap;
  resourceStates resourceState;
  BufferDesc bufferDesc;
  unsigned char* mappedBuffer;
  unsigned int bufferSize;

};

#endif