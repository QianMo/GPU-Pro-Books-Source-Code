//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------
#include "CPUT_DX11.h"
#include "CPUTMeshDX11.h"
#include "CPUTMaterialDX11.h"
#include "CPUTRenderParamsDX.h"
#include "CPUTBufferDX11.h"

//-----------------------------------------------------------------------------
CPUTMeshDX11::CPUTMeshDX11():
    mVertexStride(0),
    mVertexBufferOffset(0),
    mVertexCount(0),
    mpVertexBuffer(NULL),
    mpVertexBufferForSRVDX(NULL),
    mpVertexBufferForSRV(NULL),
    mpVertexView(NULL),
    mpStagingVertexBuffer(NULL),
    mVertexBufferMappedType(CPUT_MAP_UNDEFINED),
    mIndexBufferMappedType(CPUT_MAP_UNDEFINED),
    mIndexCount(0),
    mIndexBufferFormat(DXGI_FORMAT_UNKNOWN),
    mpIndexBuffer(NULL),
    mpStagingIndexBuffer(NULL),
    mpInputLayout(NULL),
    mpShadowInputLayout(NULL),
    mNumberOfInputLayoutElements(0),
    mpLayoutDescription(NULL)
{
}

//-----------------------------------------------------------------------------
CPUTMeshDX11::~CPUTMeshDX11()
{
    ClearAllObjects();
}

//-----------------------------------------------------------------------------
void CPUTMeshDX11::ClearAllObjects()
{
    SAFE_RELEASE(mpStagingIndexBuffer);
    SAFE_RELEASE(mpIndexBuffer);
    SAFE_RELEASE(mpStagingVertexBuffer);
    SAFE_RELEASE(mpVertexBuffer);
    SAFE_RELEASE(mpVertexBufferForSRVDX);
    SAFE_RELEASE(mpVertexBufferForSRV);
    SAFE_RELEASE(mpVertexView);
    SAFE_RELEASE(mpInputLayout);
    SAFE_RELEASE(mpShadowInputLayout);

    SAFE_DELETE_ARRAY(mpLayoutDescription);
}

// Create the DX vertex/index buffers and D3D11_INPUT_ELEMENT_DESC
//-----------------------------------------------------------------------------
CPUTResult CPUTMeshDX11::CreateNativeResources(
    CPUTModel       *pModel,
    UINT             meshIdx,
    int              vertexDataInfoArraySize,
    CPUTBufferInfo  *pVertexDataInfo,
    void            *pVertexData,
    CPUTBufferInfo  *pIndexDataInfo,
    void            *pIndexData
){

    CPUTResult result = CPUT_SUCCESS;
    HRESULT hr;

    ID3D11Device *pD3dDevice = CPUT_DX11::GetDevice();

    // Release the layout, offset, stride, and vertex buffer structure objects
    ClearAllObjects();

    // allocate the layout, offset, stride, and vertex buffer structure objects
    mpLayoutDescription = new D3D11_INPUT_ELEMENT_DESC[vertexDataInfoArraySize+1];

    // Create the index buffer
    D3D11_SUBRESOURCE_DATA resourceData;
    if(NULL!=pIndexData)
    {
        mIndexCount = pIndexDataInfo->mElementCount;

        // set the data format info
        ZeroMemory( &mIndexBufferDesc, sizeof(mIndexBufferDesc) );
        mIndexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        mIndexBufferDesc.ByteWidth = mIndexCount * pIndexDataInfo->mElementSizeInBytes;
        mIndexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
        mIndexBufferDesc.CPUAccessFlags = 0;  // default to no cpu access for speed

        // create the buffer
        ZeroMemory( &resourceData, sizeof(resourceData) );
        resourceData.pSysMem = pIndexData;
        hr = pD3dDevice->CreateBuffer( &mIndexBufferDesc, &resourceData, &mpIndexBuffer );
        ASSERT(!FAILED(hr), _L("Failed creating index buffer") );
        CPUTSetDebugName( mpIndexBuffer, _L("Index buffer") );

        // set the DX index buffer format
        mIndexBufferFormat = ConvertToDirectXFormat(pIndexDataInfo->mElementType, pIndexDataInfo->mElementComponentCount);
    }

    // set up data format info
    mVertexCount = pVertexDataInfo[0].mElementCount;

    ZeroMemory( &mVertexBufferDesc, sizeof(mVertexBufferDesc) );
    mVertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    // set the stride for one 'element' block of verts
	mVertexStride = pVertexDataInfo[vertexDataInfoArraySize-1].mOffset + pVertexDataInfo[vertexDataInfoArraySize-1].mElementSizeInBytes; // size in bytes of a single vertex block
    mVertexBufferDesc.ByteWidth = mVertexCount * mVertexStride; // size in bytes of entire buffer
    mVertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    mVertexBufferDesc.CPUAccessFlags = 0;  // default to no cpu access for speed

    // create the buffer
    ZeroMemory( &resourceData, sizeof(resourceData) );
    resourceData.pSysMem = pVertexData;
    hr = pD3dDevice->CreateBuffer( &mVertexBufferDesc, &resourceData, &mpVertexBuffer );
    ASSERT( !FAILED(hr), _L("Failed creating vertex buffer") );
    CPUTSetDebugName( mpVertexBuffer, _L("Vertex buffer") );

    // create the buffer for the shader resource view
    D3D11_BUFFER_DESC desc;
    ZeroMemory( &desc, sizeof(desc) );
    desc.Usage = D3D11_USAGE_DEFAULT;
    // set the stride for one 'element' block of verts
	mVertexStride           = pVertexDataInfo[vertexDataInfoArraySize-1].mOffset + pVertexDataInfo[vertexDataInfoArraySize-1].mElementSizeInBytes; // size in bytes of a single vertex block
    desc.ByteWidth           = mVertexCount * mVertexStride; // size in bytes of entire buffer
    desc.BindFlags           = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags      = 0;
    desc.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = mVertexStride;

    hr = pD3dDevice->CreateBuffer( &desc, &resourceData, &mpVertexBufferForSRVDX );
    ASSERT( !FAILED(hr), _L("Failed creating vertex buffer for SRV") );
    //CPUTSetDebugName( mpVertexBuffer, _L("Vertex buffer for SRV") );

    // Create the shader resource view
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;  
    ZeroMemory( &srvDesc, sizeof(srvDesc) );  
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;  
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;  
    srvDesc.Buffer.ElementOffset = 0;  
    srvDesc.Buffer.NumElements = mVertexCount;

    hr = pD3dDevice->CreateShaderResourceView( mpVertexBufferForSRVDX, &srvDesc, &mpVertexView );
    ASSERT( !FAILED(hr), _L("Failed creating vertex buffer SRV") );

    cString name = _L("@VertexBuffer") + ptoc(pModel) + itoc(meshIdx);
    mpVertexBufferForSRV = new CPUTBufferDX11( name, mpVertexBufferForSRVDX, mpVertexView );
    CPUTAssetLibrary::GetAssetLibrary()->AddBuffer( name, mpVertexBufferForSRV );

    // build the layout object
    int currentByteOffset=0;
    mNumberOfInputLayoutElements = vertexDataInfoArraySize;
    for(int ii=0; ii<vertexDataInfoArraySize; ii++)
    {
        mpLayoutDescription[ii].SemanticName  = pVertexDataInfo[ii].mpSemanticName; // string name that matches
        mpLayoutDescription[ii].SemanticIndex = pVertexDataInfo[ii].mSemanticIndex; // if we have more than one
        mpLayoutDescription[ii].Format = ConvertToDirectXFormat(pVertexDataInfo[ii].mElementType, pVertexDataInfo[ii].mElementComponentCount);
        mpLayoutDescription[ii].InputSlot = 0; // TODO: We support only a single stream now.  Support multiple streams.
        mpLayoutDescription[ii].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
        mpLayoutDescription[ii].InstanceDataStepRate = 0;
        mpLayoutDescription[ii].AlignedByteOffset = currentByteOffset;
        currentByteOffset += pVertexDataInfo[ii].mElementSizeInBytes;
    }
    // set the last 'dummy' element to null.  Not sure if this is required, as we also pass in count when using this list.
    memset( &mpLayoutDescription[vertexDataInfoArraySize], 0, sizeof(D3D11_INPUT_ELEMENT_DESC) );

    return result;
}

//-----------------------------------------------------------------------------
void CPUTMeshDX11::Draw(CPUTRenderParameters &renderParams, CPUTModel *pModel, ID3D11InputLayout *pInputLayout )
{
    // Skip empty meshes.
    if( !mIndexCount ) { return; }

    ID3D11DeviceContext *pContext = ((CPUTRenderParametersDX*)&renderParams)->mpContext;

    pContext->IASetPrimitiveTopology( mD3DMeshTopology );
    pContext->IASetVertexBuffers(0, 1, &mpVertexBuffer, &mVertexStride, &mVertexBufferOffset);
    pContext->IASetIndexBuffer(mpIndexBuffer, mIndexBufferFormat, 0);

    pContext->IASetInputLayout( pInputLayout );

    pContext->DrawIndexed( mIndexCount, 0, 0 );
}

// Sets the mesh topology, and converts it to it's DX format
//-----------------------------------------------------------------------------
void CPUTMeshDX11::SetMeshTopology(const eCPUT_MESH_TOPOLOGY meshTopology)
{
    ASSERT( meshTopology > 0 && meshTopology <= 5, _L("") );
    CPUTMesh::SetMeshTopology(meshTopology);
    // The CPUT enum has the same values as the D3D enum.  Will likely need an xlation on OpenGL.
    mD3DMeshTopology = (D3D_PRIMITIVE_TOPOLOGY)meshTopology;
}

// Translate an internal CPUT data type into it's equivalent DirectX type
//-----------------------------------------------------------------------------
DXGI_FORMAT CPUTMeshDX11::ConvertToDirectXFormat(CPUT_DATA_FORMAT_TYPE dataFormatType, int componentCount)
{
    ASSERT( componentCount>0 && componentCount<=4, _L("Invalid vertex element count.") );
    switch( dataFormatType )
    {
    case CPUT_F32:
    {
        const DXGI_FORMAT componentCountToFormat[4] = {
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_R32G32_FLOAT,
            DXGI_FORMAT_R32G32B32_FLOAT,
            DXGI_FORMAT_R32G32B32A32_FLOAT
        };
        return componentCountToFormat[componentCount-1];
    }
    case CPUT_U32:
    {
        const DXGI_FORMAT componentCountToFormat[4] = {
            DXGI_FORMAT_R32_UINT,
            DXGI_FORMAT_R32G32_UINT,
            DXGI_FORMAT_R32G32B32_UINT,
            DXGI_FORMAT_R32G32B32A32_UINT
        };
        return componentCountToFormat[componentCount-1];
        break;
    }
    case CPUT_U16:
    {
        ASSERT( 3 != componentCount, _L("Invalid vertex element count.") );
        const DXGI_FORMAT componentCountToFormat[4] = {
            DXGI_FORMAT_R16_UINT,
            DXGI_FORMAT_R16G16_UINT,
            DXGI_FORMAT_UNKNOWN, // Count of 3 is invalid for 16-bit type
            DXGI_FORMAT_R16G16B16A16_UINT
        };
        return componentCountToFormat[componentCount-1];
    }
    default:
    {
        // todo: add all the other data types you want to support
        ASSERT(0,_L("Unsupported vertex element type"));
    }
    return DXGI_FORMAT_UNKNOWN;
    }
} // CPUTMeshDX11::ConvertToDirectXFormat()

// Finds a layout that matches the materials vertex shader and the mesh's
// vertex buffer layout by using the InputLayoutCache
//-----------------------------------------------------------------------------
void CPUTMeshDX11::BindVertexShaderLayout( CPUTMaterial *pMaterial, CPUTMaterial *pShadowCastMaterial )
{
    ID3D11Device *pDevice = CPUT_DX11::GetDevice();

    if( pMaterial )
    {
        // Get the vertex layout for this shader/format comb
        // If already exists, then GetLayout() returns the existing layout for reuse.
        CPUTVertexShaderDX11 *pVertexShader = ((CPUTMaterialDX11*)pMaterial)->GetVertexShader();
	    SAFE_RELEASE(mpInputLayout);
        CPUTInputLayoutCacheDX11::GetInputLayoutCache()->GetLayout(pDevice, mpLayoutDescription, pVertexShader, &mpInputLayout);
    }
    if( pShadowCastMaterial )
    {
        CPUTVertexShaderDX11 *pVertexShader = ((CPUTMaterialDX11*)pShadowCastMaterial)->GetVertexShader();
	    SAFE_RELEASE(mpShadowInputLayout);
        CPUTInputLayoutCacheDX11::GetInputLayoutCache()->GetLayout(pDevice, mpLayoutDescription, pVertexShader, &mpShadowInputLayout);
    }
}

//-----------------------------------------------------------------------------
D3D11_MAPPED_SUBRESOURCE CPUTMeshDX11::Map(
    UINT                   count,
    ID3D11Buffer          *pBuffer,
    D3D11_BUFFER_DESC     &bufferDesc,
    ID3D11Buffer         **pStagingBuffer,
    eCPUTMapType          *pMappedType,
    CPUTRenderParameters  &params,
    eCPUTMapType           type,
    bool                   wait
)
{
    // Mapping for DISCARD requires dynamic buffer.  Create dynamic copy?
    // Could easily provide input flag.  But, where would we specify? Don't like specifying in the .set file
    // Because mapping is something the application wants to do - it isn't inherent in the data.
    // Could do Clone() and pass dynamic flag to that.
    // But, then we have two.  Could always delete the other.
    // Could support programatic flag - apply to all loaded models in the .set
    // Could support programatic flag on model.  Load model first, then load set.
    // For now, simply support CopyResource mechanism.
    HRESULT hr;
    ID3D11Device *pD3dDevice = CPUT_DX11::GetDevice();
    CPUTRenderParametersDX *pParamsDX11 = (CPUTRenderParametersDX*)&params;
    ID3D11DeviceContext *pContext = pParamsDX11->mpContext;

    if( !*pStagingBuffer )
    {
        D3D11_BUFFER_DESC desc = bufferDesc;
        // First time.  Create the staging resource
        desc.Usage = D3D11_USAGE_STAGING;
        switch( type )
        {
        case CPUT_MAP_READ:
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            desc.BindFlags = 0;
            break;
        case CPUT_MAP_READ_WRITE:
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
            desc.BindFlags = 0;
            break;
        case CPUT_MAP_WRITE:
        case CPUT_MAP_WRITE_DISCARD:
        case CPUT_MAP_NO_OVERWRITE:
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            desc.BindFlags = 0;
            break;
        };
        hr = pD3dDevice->CreateBuffer( &desc, NULL, pStagingBuffer );
        ASSERT( SUCCEEDED(hr), _L("Failed to create staging buffer") );
        CPUTSetDebugName( *pStagingBuffer, _L("Mesh Staging buffer") );
    }
    else
    {
        ASSERT( *pMappedType == type, _L("Mapping with a different CPU access than creation parameter.") );
    }

    D3D11_MAPPED_SUBRESOURCE info;
    switch( type )
    {
    case CPUT_MAP_READ:
    case CPUT_MAP_READ_WRITE:
        // TODO: Copying and immediately mapping probably introduces a stall.
        // Expose the copy externally?
        // TODO: copy only if vb has changed?
        // Copy only first time?
        // Copy the GPU version before we read from it.
        pContext->CopyResource( *pStagingBuffer, pBuffer );
        break;
    };
    hr = pContext->Map( *pStagingBuffer, wait ? 0 : D3D11_MAP_FLAG_DO_NOT_WAIT, (D3D11_MAP)type, 0, &info );
    *pMappedType = type;
    return info;
} // CPUTMeshDX11::Map()

//-----------------------------------------------------------------------------
void  CPUTMeshDX11::Unmap(
    ID3D11Buffer         *pBuffer,
    ID3D11Buffer         *pStagingBuffer,
    eCPUTMapType         *pMappedType,
    CPUTRenderParameters &params
)
{
    ASSERT( *pMappedType != CPUT_MAP_UNDEFINED, _L("Can't unmap a buffer that isn't mapped.") );

    CPUTRenderParametersDX *pParamsDX11 = (CPUTRenderParametersDX*)&params;
    ID3D11DeviceContext *pContext = pParamsDX11->mpContext;

    pContext->Unmap( pStagingBuffer, 0 );

    // If we were mapped for write, then copy staging buffer to GPU
    switch( *pMappedType )
    {
    case CPUT_MAP_READ:
        break;
    case CPUT_MAP_READ_WRITE:
    case CPUT_MAP_WRITE:
    case CPUT_MAP_WRITE_DISCARD:
    case CPUT_MAP_NO_OVERWRITE:
        pContext->CopyResource( pBuffer, pStagingBuffer );
        break;
    };

    // *pMappedType = CPUT_MAP_UNDEFINED; // TODO: Just commented this out.  Otherwise, assert in Map() fails.  Verify.
} // CPUTMeshDX11::Unmap()


//-----------------------------------------------------------------------------
D3D11_MAPPED_SUBRESOURCE CPUTMeshDX11::MapVertices( CPUTRenderParameters &params, eCPUTMapType type, bool wait )
{
    return Map(
        mVertexCount,
        mpVertexBuffer,
        mVertexBufferDesc,
       &mpStagingVertexBuffer,
       &mVertexBufferMappedType,
        params,
        type,
        wait
    );
}

//-----------------------------------------------------------------------------
D3D11_MAPPED_SUBRESOURCE CPUTMeshDX11::MapIndices( CPUTRenderParameters &params, eCPUTMapType type, bool wait )
{
    return Map(
        mIndexCount,
        mpIndexBuffer,
        mIndexBufferDesc,
       &mpStagingIndexBuffer,
       &mIndexBufferMappedType,
        params,
        type,
        wait
    );
}

//-----------------------------------------------------------------------------
void CPUTMeshDX11::UnmapVertices( CPUTRenderParameters &params )
{
    Unmap( mpVertexBuffer, mpStagingVertexBuffer, &mVertexBufferMappedType, params );
}

//-----------------------------------------------------------------------------
void CPUTMeshDX11::UnmapIndices( CPUTRenderParameters &params )
{
    Unmap( mpIndexBuffer, mpStagingIndexBuffer, &mIndexBufferMappedType, params );
}
