#ifndef DX12_IRESOURCE_H
#define DX12_IRESOURCE_H

// DX12_IResource
//
class DX12_IResource
{
public:
  virtual ID3D12Resource* GetResource() const = 0;

  virtual ID3D12Resource* GetUploadHeap() const = 0;

  virtual void SetResourceState(resourceStates resourceState) = 0;

  virtual resourceStates GetResourceState() const = 0;

  virtual UINT GetNumSubresources() const = 0;

};

class DX12_UploadHelper
{
public:
  DX12_UploadHelper(DX12_IResource *resource) :
    layouts(nullptr),
    rowSizesInBytes(nullptr),
    numRows(nullptr)
  {
    assert(resource != nullptr);
    this->resource = resource;
  }

  ~DX12_UploadHelper()
  {
    SAFE_DELETE_ARRAY(layouts);
    SAFE_DELETE_ARRAY(rowSizesInBytes);
    SAFE_DELETE_ARRAY(numRows);
  }

  template <UINT numSubresources>
  bool CopySubresources(const D3D12_SUBRESOURCE_DATA *srcData)
  {
    assert(srcData != nullptr);

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layoutsStack[numSubresources];
    UINT64 rowSizesInBytesStack[numSubresources];
    UINT numRowsStack[numSubresources];

    if(!CopySubresources(layoutsStack, rowSizesInBytesStack, numRowsStack, numSubresources, srcData))
      return false;

    return true;
  }

  bool CopySubresources(const D3D12_SUBRESOURCE_DATA *srcData, UINT numSubresources)
  {
    assert((layouts == nullptr) && (rowSizesInBytes == nullptr) && (numRows == nullptr) && (srcData != nullptr));

    layouts = new D3D12_PLACED_SUBRESOURCE_FOOTPRINT[numSubresources];
    rowSizesInBytes = new UINT64[numSubresources];
    numRows = new UINT[numSubresources];
    if((!layouts) || (!rowSizesInBytes) || (!numRows))
      return false;

    if(!CopySubresources(layouts, rowSizesInBytes, numRows, numSubresources, srcData))
      return false;

    return true;
  }

private:
  bool CopySubresources(D3D12_PLACED_SUBRESOURCE_FOOTPRINT *layoutsIn, UINT64 *rowSizesInBytesIn, UINT *numRowsIn, UINT numSubresources,
    const D3D12_SUBRESOURCE_DATA *srcData)
  {
    ID3D12Resource *destResource = resource->GetResource();
    assert(destResource != nullptr);
    D3D12_RESOURCE_DESC desc = destResource->GetDesc();

    ID3D12Device *device;
    destResource->GetDevice(__uuidof(*device), reinterpret_cast<void**>(&device));
    device->GetCopyableFootprints(&desc, 0, numSubresources, 0, layoutsIn, numRowsIn, rowSizesInBytesIn, nullptr);
    device->Release();

    BYTE *dstData;
    ID3D12Resource *uploadHeap = resource->GetUploadHeap();
    assert(uploadHeap != nullptr);
    if(FAILED(uploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&dstData))))
    {
      return false;
    }
    for(UINT i=0; i<numSubresources; i++)
    {
      D3D12_MEMCPY_DEST destData = { dstData + layoutsIn[i].Offset, layoutsIn[i].Footprint.RowPitch, layoutsIn[i].Footprint.RowPitch * numRowsIn[i] };
      MemcpySubresource(&destData, &srcData[i], (SIZE_T)rowSizesInBytesIn[i], numRowsIn[i], layoutsIn[i].Footprint.Depth);
    }
    uploadHeap->Unmap(0, nullptr);

    return true;
  }

  DX12_IResource *resource;
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT *layouts;
  UINT64 *rowSizesInBytes;
  UINT *numRows;

};

#endif