/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_BUFFER_DX11
#define BE_GRAPHICS_BUFFER_DX11

#include "beGraphics.h"
#include "../beBuffer.h"
#include <beCore/beWrapper.h>
#include <D3D11.h>
#include <lean/smart/com_ptr.h>
#include <lean/tags/noncopyable.h>

namespace beGraphics
{

namespace DX11
{

/// Creates a structured buffer.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11Buffer, true> CreateStructuredBuffer(ID3D11Device *device, uint4 bindFlags, uint4 size, uint4 count = 1, uint4 flags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, const void *pInitialData = nullptr);
/// Creates a constant buffer.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11Buffer, true> CreateConstantBuffer(ID3D11Device *device, uint4 size, uint4 count = 1, const void *pInitialData = nullptr);
/// Clones the given buffer (not cloning contents!).
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11Buffer, true> CloneBuffer(ID3D11Buffer &buffer, const void *pInitialData = nullptr);
/// Creates a staging buffer.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11Buffer, true> CreateStagingBuffer(ID3D11Device *device, uint4 size, uint4 count = 1, uint4 cpuAccess = D3D11_CPU_ACCESS_READ, const void *pInitialData = nullptr);
/// Creates a staging buffer matching the given buffer.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11Buffer, true> CreateStagingBuffer(ID3D11Device *device, ID3D11Buffer *buffer, uint4 cpuAccess = D3D11_CPU_ACCESS_READ, const void *pInitialData = nullptr);
/// Copies data from the given source buffer to the given destination buffer.
BE_GRAPHICS_DX11_API void CopyBuffer(ID3D11DeviceContext *context, ID3D11Buffer *dest, uint4 destOffset, ID3D11Buffer *src, uint4 srcOffset, uint4 srcCount);
/// Copies the given data to the given destination buffer.
BE_GRAPHICS_DX11_API void WriteBuffer(ID3D11DeviceContext *context, ID3D11Buffer *dest, uint4 destOffset, const void *data, uint4 srcOffset, uint4 srcCount);
/// Copies the given data to the given destination buffer.
BE_GRAPHICS_DX11_API bool WriteBufferByMap(ID3D11DeviceContext *context, ID3D11Buffer *dest, uint4 destOffset, const void *data, uint4 srcOffset, uint4 srcCount);

/// Creates an unordered access view.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11UnorderedAccessView, true> CreateUAV(ID3D11Buffer *buffer, const D3D11_UNORDERED_ACCESS_VIEW_DESC *pDesc = nullptr);
/// Creates a shader resource view.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11ShaderResourceView, true> CreateSRV(ID3D11Buffer *buffer, const D3D11_SHADER_RESOURCE_VIEW_DESC *pDesc = nullptr);

/// Creates an unordered access view.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11UnorderedAccessView, true> CreateUAV(ID3D11Buffer *buffer, DXGI_FORMAT fmt, uint4 flags, uint4 offset, uint4 count);
/// Creates a shader resource view.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11ShaderResourceView, true> CreateSRV(ID3D11Buffer *buffer, DXGI_FORMAT fmt, uint4 flags, uint4 offset, uint4 count);

/// Creates a counting view.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11UnorderedAccessView, true> CreateCountingUAV(ID3D11Buffer *buffer, uint4 flags = 0, uint4 offset = 0, uint4 count = -1);
/// Creates a raw view.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11UnorderedAccessView, true> CreateRawUAV(ID3D11Buffer *buffer, uint4 flags = 0, uint4 offset = 0, uint4 count = -1);
/// Creates a raw view.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11ShaderResourceView, true> CreateRawSRV(ID3D11Buffer *buffer, uint4 flags = 0, uint4 offset = 0, uint4 count = -1);

/// Creates a buffer according to the given description.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11Buffer, true> CreateBuffer(const D3D11_BUFFER_DESC &desc, const void *pInitialData, ID3D11Device *pDevice);
/// Updates the given unstructured buffer with the given data.
BE_GRAPHICS_DX11_API void PartialUpdate(ID3D11DeviceContext *pDeviceContext, ID3D11Buffer *pBuffer, uint4 offset, uint4 endOffset, const void *pData, uint4 sourceOffset = 0);
/// Maps this buffer to allow for CPU access.
BE_GRAPHICS_DX11_API bool Map(ID3D11DeviceContext *pDeviceContext, ID3D11Buffer *pBuffer, void *&data, D3D11_MAP map, uint4 flags = 0);
/// Unmaps this buffer to allow for GPU access.
BE_GRAPHICS_DX11_API void Unmap(ID3D11DeviceContext *pDeviceContext, ID3D11Buffer *pBuffer);

/// Gets the description of the given buffer.
BE_GRAPHICS_DX11_API D3D11_BUFFER_DESC GetDesc(ID3D11Buffer &buffer);

/// Gets data from the given buffer.
BE_GRAPHICS_DX11_API bool ReadBufferData(ID3D11DeviceContext *context, ID3D11Buffer *buffer, void *bytes, uint4 byteCount, uint4 resourceOffset = 0);
/// Gets data from the given buffer using a TEMPORARY staging buffer. SLOW!
BE_GRAPHICS_DX11_API bool DebugFetchBufferData(ID3D11DeviceContext *context, ID3D11Buffer *buffer, void *bytes, uint4 byteCount, uint4 resourceOffset = 0);

/// Buffer wrapper.
class Buffer : public beCore::IntransitiveWrapper<ID3D11Buffer, Buffer>, public beGraphics::Buffer
{
private:
	lean::com_ptr<ID3D11Buffer> m_pBuffer;

public:
	/// Constructor.
	BE_GRAPHICS_DX11_API Buffer(const D3D11_BUFFER_DESC &desc, const void *pInitialData, ID3D11Device *pDevice);
	/// Constructor.
	BE_GRAPHICS_DX11_API Buffer(ID3D11Buffer *pBuffer);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~Buffer();

	/// Maps this buffer to allow for CPU access.
	LEAN_INLINE bool Map(ID3D11DeviceContext *pDeviceContext, void *&data, D3D11_MAP map, uint4 flags = 0)
	{
		return DX11::Map(pDeviceContext, m_pBuffer, data, map, flags);
	}
	/// Unmaps this buffer to allow for GPU access.
	LEAN_INLINE void Unmap(ID3D11DeviceContext *pDeviceContext)
	{
		DX11::Unmap(pDeviceContext, m_pBuffer);
	}

	/// Gets the implementation identifier.
	LEAN_INLINE ImplementationID GetImplementationID() const { return DX11Implementation; }

	/// Gets the D3D buffer.
	LEAN_INLINE ID3D11Buffer*const& GetInterface() const { return m_pBuffer.get(); }
	/// Gets the D3D buffer.
	LEAN_INLINE ID3D11Buffer*const& GetBuffer() const { return m_pBuffer.get(); }
};

template <> struct ToImplementationDX11<beGraphics::Buffer> { typedef Buffer Type; };

/// Scoped mapped buffer.
struct MappedBuffer : public lean::noncopyable
{
private:
	ID3D11Buffer *m_pBuffer;
	ID3D11DeviceContext *m_pContext;
	void *m_data;

public:
	/// Maps the given buffer in the given device context.
	LEAN_INLINE MappedBuffer(ID3D11Buffer *pBuffer, ID3D11DeviceContext *pDeviceContext, D3D11_MAP map, uint4 flags = 0)
		:  m_pBuffer(pBuffer),
		m_pContext(pDeviceContext),
		m_data(nullptr)
	{
		Map(m_pContext, m_pBuffer, m_data, map, flags);
	}
	/// Unmaps the managed buffer.
	LEAN_INLINE ~MappedBuffer()
	{
		Unmap();
	}

	/// Unmaps the managed buffer.
	LEAN_INLINE void Unmap()
	{
		if (m_data)
		{
			m_data = nullptr;
			DX11::Unmap(m_pContext, m_pBuffer);
		}
	}

	/// Gets the mapped data.
	LEAN_INLINE void* Data() const { return m_data; }
};

} // namespace

} // namespace

#endif