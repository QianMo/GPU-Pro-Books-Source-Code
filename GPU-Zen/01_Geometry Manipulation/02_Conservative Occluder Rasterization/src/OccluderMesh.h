#pragma once

#include "Graphics/Graphics.h"

#include <vector>

class COccluderMesh
{
private:
    ID3D12Resource* m_VertexBuffer;
    ID3D12Resource* m_VertexSrvBuffer;
    ID3D12Resource* m_VertexBufferUpload;
    ID3D12Resource* m_IndexBuffer;
    ID3D12Resource* m_IndexBufferUpload;
    ID3D12Resource* m_IndexBufferLine;
    ID3D12Resource* m_IndexBufferLineUpload;
    ID3D12Resource* m_IndexBufferAdj;
    ID3D12Resource* m_IndexBufferAdjUpload;

    D3D12_VERTEX_BUFFER_VIEW m_VertexBufferView;
    D3D12_INDEX_BUFFER_VIEW m_IndexBufferView;
    D3D12_INDEX_BUFFER_VIEW m_IndexBufferLineView;
    D3D12_INDEX_BUFFER_VIEW m_IndexBufferAdjView;

    NGraphics::SDescriptorHandle m_VertexSrvBufferSrv;
    NGraphics::SDescriptorHandle m_IndexBufferAdjSrv;

    UINT m_IndexCount;
    UINT m_IndexLineCount;
    UINT m_IndexAdjCount;
    UINT m_FaceCount;

public:
    COccluderMesh();

    virtual void Create(
        ID3D12Device* device,
        NGraphics::CCommandContext* graphics_context,
        NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] ) = 0;
    void Destroy();

    void Draw( ID3D12GraphicsCommandList* command_list, UINT instance_count = 1 );
    void DrawLine( ID3D12GraphicsCommandList* command_list, UINT instance_count = 1 );
    void DrawAdj( ID3D12GraphicsCommandList* command_list, UINT instance_count = 1 );

    D3D12_VERTEX_BUFFER_VIEW GetVertexBufferView() const;
    D3D12_INDEX_BUFFER_VIEW GetIndexBufferView() const;

    NGraphics::SDescriptorHandle GetVertexSrvBufferSrv() const;
    NGraphics::SDescriptorHandle GetIndexBufferAdjSrv() const;

    const UINT GetIndexCount() const;
    const UINT GetFaceCount() const;

protected:
    void Create(
        ID3D12Device* device,
        NGraphics::CCommandContext* graphics_context,
        NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ],
        std::vector< DirectX::XMFLOAT3 >* vertices,
        std::vector< UINT >* indices,
        std::vector< UINT >* indices_line,
        std::vector< UINT >* indices_adj );
};

class CCubeOccluderMesh : public COccluderMesh
{
public:
    void Create( ID3D12Device* device,
                 NGraphics::CCommandContext* graphics_context,
                 NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] );
};

class CCylinderOccluderMesh : public COccluderMesh
{
private:
    const UINT m_CylinderSliceCount = 16;

public:
    void Create( ID3D12Device* device,
                 NGraphics::CCommandContext* graphics_context,
                 NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] );
};