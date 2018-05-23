/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/DX11/beBuffer.h"
#include "beGraphics/DX11/beDevice.h"
#include "beGraphics/DX11/beFormat.h"
#include "beGraphics/DX/beError.h"

namespace beGraphics
{

namespace DX11
{

/// Creates a structured buffer.
lean::com_ptr<ID3D11Buffer, true> CreateStructuredBuffer(ID3D11Device *device, uint4 bindFlags, uint4 size, uint4 count, uint4 flags, const void *pInitialData)
{
	D3D11_BUFFER_DESC desc;
	desc.ByteWidth = size * count;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = bindFlags;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = flags;
	desc.StructureByteStride = size;

	return CreateBuffer(desc, pInitialData, device);
}

/// Creates a constant buffer.
lean::com_ptr<ID3D11Buffer, true> CreateConstantBuffer(ID3D11Device *device, uint4 size, uint4 count, const void *pInitialData)
{
	return CreateStructuredBuffer(device, D3D11_BIND_CONSTANT_BUFFER, size, count, 0, pInitialData);
}

// Clones the given buffer (not cloning contents!).
lean::com_ptr<ID3D11Buffer, true> CloneBuffer(ID3D11Buffer &buffer, const void *pInitialData)
{
	return CreateBuffer(GetDesc(buffer), pInitialData, GetDevice(buffer));
}

/// Creates a staging buffer.
lean::com_ptr<ID3D11Buffer, true> CreateStagingBuffer(ID3D11Device *device, uint4 size, uint4 count, uint4 cpuAccess, const void *pInitialData)
{
	D3D11_BUFFER_DESC desc;
	desc.ByteWidth = size * count;
	desc.Usage = D3D11_USAGE_STAGING;
	desc.BindFlags = 0;
	desc.CPUAccessFlags = cpuAccess;
	desc.MiscFlags = 0;
	desc.StructureByteStride = size;

	return CreateBuffer(desc, pInitialData, device);
}

/// Creates a staging buffer matching the given buffer.
lean::com_ptr<ID3D11Buffer, true> CreateStagingBuffer(ID3D11Device *device, ID3D11Buffer *buffer, uint4 cpuAccess, const void *pInitialData)
{
	D3D11_BUFFER_DESC desc = GetDesc(*buffer);
	desc.Usage = D3D11_USAGE_STAGING;
	desc.BindFlags = 0;
	desc.CPUAccessFlags = cpuAccess;
	// desc.MiscFlags &= D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
	desc.MiscFlags = 0;

	return CreateBuffer(desc, pInitialData, device);
}

// Creates an unordered access view.
lean::com_ptr<ID3D11UnorderedAccessView, true> CreateUAV(ID3D11Buffer *buffer, const D3D11_UNORDERED_ACCESS_VIEW_DESC *pDesc)
{
	lean::com_ptr<ID3D11UnorderedAccessView> uav;

	BE_THROW_DX_ERROR_MSG(
		GetDevice(*buffer)->CreateUnorderedAccessView(buffer, pDesc, uav.rebind()),
		"ID3D11Device::CreateUnorderedAccessView()" );

	return uav.transfer();
}

// Creates a shader resource view.
lean::com_ptr<ID3D11ShaderResourceView, true> CreateSRV(ID3D11Buffer *buffer, const D3D11_SHADER_RESOURCE_VIEW_DESC *pDesc)
{
	lean::com_ptr<ID3D11ShaderResourceView> srv;

	BE_THROW_DX_ERROR_MSG(
		GetDevice(*buffer)->CreateShaderResourceView(buffer, pDesc, srv.rebind()),
		"ID3D11Device::CreateShaderResourceView()" );

	return srv.transfer();
}

// Creates an unordered access view.
lean::com_ptr<ID3D11UnorderedAccessView, true> CreateUAV(ID3D11Buffer *buffer, DXGI_FORMAT fmt, uint4 flags, uint4 offset, uint4 count)
{
	lean::com_ptr<ID3D11UnorderedAccessView> uav;

	if (fmt != DXGI_FORMAT_UNKNOWN || flags != 0 || offset != 0 || count != -1)
	{
		D3D11_BUFFER_DESC bufferDesc = GetDesc(*buffer);
		
		// Unstructured buffers lack valid stride
		if (bufferDesc.StructureByteStride == 0)
			bufferDesc.StructureByteStride = SizeofFormat(fmt);

		D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
		desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		
		if (flags & D3D11_BUFFER_UAV_FLAG_RAW)
		{
			desc.Format = DXGI_FORMAT_R32_TYPELESS;
			desc.Buffer.FirstElement = offset / sizeof(UINT);
			desc.Buffer.NumElements = min(bufferDesc.ByteWidth, count) / sizeof(UINT);
		}
		else
		{
			desc.Format = fmt;
			desc.Buffer.FirstElement = offset;
			desc.Buffer.NumElements = min(bufferDesc.ByteWidth / bufferDesc.StructureByteStride, count);
		}

		desc.Buffer.Flags = flags;

		uav = CreateUAV(buffer, &desc);
	}
	else
		uav = CreateUAV(buffer, nullptr);

	return uav.transfer();
}

// Creates a shader resource view.
lean::com_ptr<ID3D11ShaderResourceView, true> CreateSRV(ID3D11Buffer *buffer, DXGI_FORMAT fmt, uint4 flags, uint4 offset, uint4 count)
{
	lean::com_ptr<ID3D11ShaderResourceView> srv;

	if (fmt != DXGI_FORMAT_UNKNOWN || flags != 0 || offset != 0 || count != -1)
	{
		D3D11_BUFFER_DESC bufferDesc = GetDesc(*buffer);
		
		// Unstructured buffers lack valid stride
		if (bufferDesc.StructureByteStride == 0)
			bufferDesc.StructureByteStride = SizeofFormat(fmt);

		D3D11_SHADER_RESOURCE_VIEW_DESC desc;

		if (flags != 0)
		{
			desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
			
			if (flags & D3D11_BUFFEREX_SRV_FLAG_RAW)
			{
				desc.Format = DXGI_FORMAT_R32_TYPELESS;
				desc.BufferEx.FirstElement = offset / sizeof(UINT);
				desc.BufferEx.NumElements = min(bufferDesc.ByteWidth, count) / sizeof(UINT);
			}
			else
			{
				desc.Format = fmt;
				desc.BufferEx.FirstElement = offset;
				desc.BufferEx.NumElements = min(bufferDesc.ByteWidth / bufferDesc.StructureByteStride, count);
			}
		
			desc.BufferEx.Flags = flags;
		}
		else
		{
			desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
			desc.Format = fmt;
			desc.Buffer.FirstElement = offset;
			desc.Buffer.NumElements = min(bufferDesc.ByteWidth / bufferDesc.StructureByteStride, count);
		}

		srv = CreateSRV(buffer, &desc);
	}
	else
		srv = CreateSRV(buffer, nullptr);

	return srv.transfer();
}

/// Creates a counting view.
lean::com_ptr<ID3D11UnorderedAccessView, true> CreateCountingUAV(ID3D11Buffer *buffer, uint4 flags, uint4 offset, uint4 count)
{
	return CreateUAV(buffer, DXGI_FORMAT_UNKNOWN, D3D11_BUFFER_UAV_FLAG_COUNTER | flags, offset, count);
}

/// Creates a raw view.
lean::com_ptr<ID3D11UnorderedAccessView, true> CreateRawUAV(ID3D11Buffer *buffer, uint4 flags, uint4 offset, uint4 count)
{
	return CreateUAV(buffer, DXGI_FORMAT_UNKNOWN, D3D11_BUFFER_UAV_FLAG_RAW | flags, offset, count);
}

/// Creates a raw view.
lean::com_ptr<ID3D11ShaderResourceView, true> CreateRawSRV(ID3D11Buffer *buffer, uint4 flags, uint4 offset, uint4 count)
{
	return CreateSRV(buffer, DXGI_FORMAT_UNKNOWN, D3D11_BUFFEREX_SRV_FLAG_RAW | flags, offset, count);
}

// Creates a buffer according to the given description.
lean::com_ptr<ID3D11Buffer, true> CreateBuffer(const D3D11_BUFFER_DESC &desc, const void *pInitialData, ID3D11Device *pDevice)
{
	lean::com_ptr<ID3D11Buffer> pBuffer;

	D3D11_SUBRESOURCE_DATA initialData = { pInitialData };
	BE_THROW_DX_ERROR_MSG(
		pDevice->CreateBuffer(&desc, pInitialData ? &initialData : nullptr, pBuffer.rebind()),
		"ID3D11Device::CreateBuffer()");

	return pBuffer.transfer();
}

// Gets the description of the given buffer.
D3D11_BUFFER_DESC GetDesc(ID3D11Buffer &buffer)
{
	D3D11_BUFFER_DESC desc;
	buffer.GetDesc(&desc);
	return desc;
}

// Constructor.
Buffer::Buffer(const D3D11_BUFFER_DESC &desc, const void *pInitialData, ID3D11Device *pDevice)
	: m_pBuffer( CreateBuffer(desc, pInitialData, pDevice) )
{
}

// Constructor.
Buffer::Buffer(ID3D11Buffer *pBuffer)
	: m_pBuffer(pBuffer)
{
	LEAN_ASSERT(m_pBuffer != nullptr);
}

// Destructor.
Buffer::~Buffer()
{
}

// Updates the given unstructured buffer with the given data.
void PartialUpdate(ID3D11DeviceContext *pDeviceContext, ID3D11Buffer *pBuffer, uint4 offset, uint4 endOffset, const void *pData, uint4 sourceOffset)
{
	LEAN_ASSERT(pDeviceContext != nullptr);
	LEAN_ASSERT(pBuffer != nullptr);
	LEAN_ASSERT(pData != nullptr);

	D3D11_BOX destBox = { offset, 0, 0, endOffset, 1, 1 };

	pDeviceContext->UpdateSubresource(
		pBuffer, 0, &destBox,
		reinterpret_cast<const char*>(&pData) + sourceOffset,
		0, 0);
}

// Maps this buffer to allow for CPU access.
bool Map(ID3D11DeviceContext *pDeviceContext, ID3D11Buffer *pBuffer, void *&data, D3D11_MAP map, uint4 flags)
{
	LEAN_ASSERT(pDeviceContext != nullptr);
	LEAN_ASSERT(pBuffer != nullptr);
	
	D3D11_MAPPED_SUBRESOURCE mappedResource;

	bool bSuccess = BE_LOG_DX_ERROR_MSG(
		pDeviceContext->Map(
			pBuffer, 0,
			map, flags,
			&mappedResource),
		"ID3D11DeviceContext::Map()");

	data = (bSuccess) ? mappedResource.pData : nullptr;
	return bSuccess;
}

// Unmaps this buffer to allow for GPU access.
void Unmap(ID3D11DeviceContext *pDeviceContext, ID3D11Buffer *pBuffer)
{
	LEAN_ASSERT(pDeviceContext != nullptr);
	LEAN_ASSERT(pBuffer != nullptr);

	pDeviceContext->Unmap(pBuffer, 0);
}

// Gets data from the given buffer.
bool ReadBufferData(ID3D11DeviceContext *context, ID3D11Buffer *buffer, void *bytes, uint4 byteCount, uint4 resourceOffset)
{
	D3D11_MAPPED_SUBRESOURCE mapped;

	if (BE_LOG_DX_ERROR_MSG(
			context->Map(
					buffer, 0,
					D3D11_MAP_READ, 0,
					&mapped
				),
			"ID3D11DeviceContext::Map()" )
		)
	{
		memcpy(bytes, static_cast<char*>(mapped.pData) + resourceOffset, byteCount);

		context->Unmap(buffer, 0);

		return true;
	}
	else
		return false;
}

// Gets data from the given buffer using a TEMPORARY staging buffer. SLOW!
bool DebugFetchBufferData(ID3D11DeviceContext *context, ID3D11Buffer *buffer, void *bytes, uint4 byteCount, uint4 resourceOffset)
{
	bool result;

	try
	{
		lean::com_ptr<ID3D11Buffer> stagingBuffer = CreateStagingBuffer(GetDevice(*buffer), buffer);

		context->CopyResource(stagingBuffer, buffer);

		result = ReadBufferData(context, stagingBuffer, bytes, byteCount, resourceOffset);
	}
	catch (const std::runtime_error &)
	{
		result = false;
	}

	return result;
}

// Copies data from the given source buffer to the given destination buffer.
void CopyBuffer(ID3D11DeviceContext *context, ID3D11Buffer *dest, uint4 destOffset, ID3D11Buffer *src, uint4 srcOffset, uint4 srcCount)
{
	D3D11_BOX box = { srcOffset, 0, 0, srcOffset + srcCount, 1, 1 };
	context->CopySubresourceRegion(dest, 0, destOffset, 0, 0, src, 0, &box);
}

// Copies the given data to the given destination buffer.
void WriteBuffer(ID3D11DeviceContext *context, ID3D11Buffer *dest, uint4 destOffset, const void *data, uint4 srcOffset, uint4 srcCount)
{
	D3D11_BOX box = { destOffset, 0, 0, destOffset + srcCount, 1, 1 };
	context->UpdateSubresource(dest, 0, &box, static_cast<const char*>(data) + srcOffset, 0, 0);
}

// Copies the given data to the given destination buffer.
bool WriteBufferByMap(ID3D11DeviceContext *context, ID3D11Buffer *dest, uint4 destOffset, const void *data, uint4 srcOffset, uint4 srcCount)
{
	D3D11_MAPPED_SUBRESOURCE mapped;

	if (BE_LOG_DX_ERROR_MSG(
			context->Map(
					dest, 0,
					D3D11_MAP_WRITE, 0,
					&mapped
				),
			"ID3D11DeviceContext::Map()" )
		)
	{
		memcpy(static_cast<char*>(mapped.pData) + destOffset, static_cast<const char*>(data) + srcOffset, srcCount);

		context->Unmap(dest, 0);
		return true;
	}
	else
		return false;
}

} // namespace

} // namespace