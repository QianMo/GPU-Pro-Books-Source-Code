/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#include "beGraphicsInternal/stdafx.h"

#define BE_GRAPHICS_TEXTURE_DX11_INSTANTIATE

#include "beGraphics/DX11/beTexture.h"
#include "beGraphics/DX11/beFormat.h"
#include "beGraphics/DX11/beDevice.h"
#include "beGraphics/DX/beError.h"
#include <DirectXTex.h>
#include <lean/strings/conversions.h>

namespace beGraphics
{

namespace DX11
{

/// Finds the SRGB format for the given format, if available. Returns unchanged format, otherwise.
DXGI_FORMAT ToSRGB(DXGI_FORMAT fmt)
{
	switch (fmt)
	{
	case DXGI_FORMAT_R8G8B8A8_UNORM:
		return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
	case DXGI_FORMAT_BC1_UNORM:
		return DXGI_FORMAT_BC1_UNORM_SRGB;
	case DXGI_FORMAT_BC2_UNORM:
		return DXGI_FORMAT_BC2_UNORM_SRGB;
	case DXGI_FORMAT_BC3_UNORM:
		return DXGI_FORMAT_BC3_UNORM_SRGB;
	case DXGI_FORMAT_BC7_UNORM:
		return DXGI_FORMAT_BC7_UNORM_SRGB;
	case DXGI_FORMAT_B8G8R8A8_UNORM:
		return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
	case DXGI_FORMAT_B8G8R8X8_UNORM:
		return DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
	}
	return fmt;
}

namespace
{

// Ripped from DirectXTex internals
inline static bool ispow2(_In_ size_t x)
{
	return ((x != 0) && !(x & (x - 1)));
}

// Ripped from DirectXTex internals
size_t CountMips( _In_ size_t width, _In_ size_t height)
{
	size_t mipLevels = 1;

	while ( height > 1 || width > 1 )
	{
		if ( height > 1 )
			height >>= 1;

		if ( width > 1 )
			width >>= 1;

		++mipLevels;
	}
	
	return mipLevels;
}

// Ripped from DirectXTex internals
size_t _CountMips3D( _In_ size_t width, _In_ size_t height, _In_ size_t depth)
{
	size_t mipLevels = 1;

	while ( height > 1 || width > 1 || depth > 1 )
	{
		if ( height > 1 )
			height >>= 1;

		if ( width > 1 )
			width >>= 1;

		if ( depth > 1 )
			depth >>= 1;

		++mipLevels;
	}
	
	return mipLevels;
}

size_t CountMips3D(_In_ size_t width, _In_ size_t height, _In_ size_t depth)
{
	return ispow2(width) && ispow2(height) && ispow2(depth)
		? _CountMips3D(width, height, depth)
		: 1;
}

// Loads a texture from the given memory.
DirectX::ScratchImage& ConvertTexture(DirectX::ScratchImage &image, DirectX::ScratchImage &scratchImage, const TextureDesc *pDesc, bool bSRGB)
{
	DirectX::ScratchImage *result = &image;

	// Fix input meta data
	if (bSRGB)
		image.OverrideFormat( DirectX::MakeSRGB(image.GetMetadata().format) );

	DirectX::TexMetadata srcMetadata = image.GetMetadata();
	DirectX::TexMetadata destMetadata = srcMetadata;

	// Apply custom description
	if (pDesc)
	{
		if (pDesc->Width != 0)
			destMetadata.width = pDesc->Width;
		if (pDesc->Height != 0)
			destMetadata.height = pDesc->Height;
		if (pDesc->Depth != 0)
			destMetadata.depth = pDesc->Depth;
		if (pDesc->Format != Format::Unknown)
			destMetadata.format = ToAPI(pDesc->Format);
		if (pDesc->MipLevels != -1)
			destMetadata.mipLevels = pDesc->MipLevels;
	}

	// Default mip creation behavior
	if (!pDesc || pDesc->MipLevels == 0)
		// Create ALL mip levels
		destMetadata.mipLevels = (destMetadata.depth <= 1)
			? CountMips(destMetadata.width, destMetadata.height)
			: CountMips3D(destMetadata.width, destMetadata.height, destMetadata.depth);

	// Check for modifications
	bool bResize = srcMetadata.width != destMetadata.width || srcMetadata.height != destMetadata.height || srcMetadata.depth != destMetadata.depth;
	bool bConvert = srcMetadata.format != destMetadata.format;
	bool bMip = srcMetadata.mipLevels != destMetadata.mipLevels;

	// Set up temporary storage
	DirectX::ScratchImage scratchImage2, scratchImage3;
	DirectX::ScratchImage &resized = (bConvert || bMip) ? scratchImage3 : scratchImage;
	DirectX::ScratchImage &converted = (bMip) ? scratchImage2 : scratchImage;
	DirectX::ScratchImage &mipped = scratchImage;

	// Apply modifications
	if (bResize)
	{
		if (srcMetadata.mipLevels > 1)
			LEAN_THROW_ERROR_MSG("Resizing unsupported for mip levels");
		if (destMetadata.depth != srcMetadata.depth)
			LEAN_THROW_ERROR_MSG("Depth resampling unsupported");

		BE_THROW_DX_ERROR_MSG(
			DirectX::Resize(result->GetImages(), result->GetImageCount(), result->GetMetadata(),
				destMetadata.width, destMetadata.height, DirectX::TEX_FILTER_CUBIC, resized),
			"DirectX::Resize()" );
		result = &resized;
	}

	if (bConvert)
	{
		BE_THROW_DX_ERROR_MSG(
			DirectX::Convert(result->GetImages(), result->GetImageCount(), result->GetMetadata(),
				destMetadata.format, DirectX::TEX_FILTER_DEFAULT, 0.0f, converted),
			"DirectX::Convert()" );
		result = &converted;
	}

	if (bMip)
	{
		if (srcMetadata.depth <= 1)
			BE_THROW_DX_ERROR_MSG(
				DirectX::GenerateMipMaps(result->GetImages(), result->GetImageCount(), result->GetMetadata(),
					DirectX::TEX_FILTER_FANT, destMetadata.mipLevels, mipped),
				"DirectX::GenerateMipMaps()" );
		else
			BE_THROW_DX_ERROR_MSG(
				DirectX::GenerateMipMaps3D(result->GetImages(), result->GetImageCount(), result->GetMetadata(),
					DirectX::TEX_FILTER_FANT, destMetadata.mipLevels, mipped),
				"DirectX::GenerateMipMaps()" );
		result = &mipped;
	}
	
	return *result;
}

} // namespace

// Loads a texture from the given file.
lean::com_ptr<ID3D11Resource, true> LoadTexture(ID3D11Device *device, const lean::utf8_ntri &fileName, const TextureDesc *pDesc, bool bSRGB)
{
	lean::com_ptr<ID3D11Resource> pTexture;

	lean::utf16_string wideFileName = lean::utf_to_utf16(fileName);
	
	DirectX::ScratchImage image;
	
	if (FAILED(DirectX::LoadFromDDSFile(wideFileName.c_str(), DirectX::DDS_FLAGS_NONE, nullptr, image)))
		BE_THROW_DX_ERROR_CTX(
			DirectX::LoadFromWICFile(wideFileName.c_str(), DirectX::WIC_FLAGS_NONE, nullptr, image),
			"DirectX::LoadFromWICFile()",
			fileName.c_str() );
		
	DirectX::ScratchImage scratchImage;
	DirectX::ScratchImage &converted = ConvertTexture(image, scratchImage, pDesc, bSRGB);

	BE_THROW_DX_ERROR_CTX(
		DirectX::CreateTexture(device, converted.GetImages(), converted.GetImageCount(), converted.GetMetadata(), pTexture.rebind()),
		"DirectX::CreateTexture()",
		fileName.c_str() );
	
	return pTexture.transfer();
}

// Loads a texture from the given memory.
lean::com_ptr<ID3D11Resource, true> LoadTexture(ID3D11Device *device, const char *data, uint4 dataLength, const TextureDesc *pDesc, bool bSRGB)
{
	lean::com_ptr<ID3D11Resource> pTexture;

	DirectX::ScratchImage image;
	
	if (FAILED(DirectX::LoadFromDDSMemory(data, dataLength, DirectX::DDS_FLAGS_NONE, nullptr, image)))
		BE_THROW_DX_ERROR_MSG(
			DirectX::LoadFromWICMemory(data, dataLength, DirectX::WIC_FLAGS_NONE, nullptr, image),
			"DirectX::LoadFromWICMemory()" );
		
	DirectX::ScratchImage scratchImage;
	DirectX::ScratchImage &converted = ConvertTexture(image, scratchImage, pDesc, bSRGB);

	BE_THROW_DX_ERROR_MSG(
		DirectX::CreateTexture(device, converted.GetImages(), converted.GetImageCount(), converted.GetMetadata(), pTexture.rebind()),
		"DirectX::CreateTexture()" );
	
	return pTexture.transfer();
}

// Creates a texture from the given texture resource.
lean::resource_ptr<Texture, true> CreateTexture(ID3D11Resource *pTextureResource)
{
	LEAN_ASSERT_NOT_NULL(pTextureResource);

	Texture *pTexture = nullptr;

	D3D11_RESOURCE_DIMENSION texDim;
	pTextureResource->GetType(&texDim);

	switch (texDim)
	{
	case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
		pTexture = new Texture1D( static_cast<ID3D11Texture1D*>(pTextureResource) );
		break;
	case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
		pTexture = new Texture2D( static_cast<ID3D11Texture2D*>(pTextureResource) );
		break;
	case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
		pTexture = new Texture3D( static_cast<ID3D11Texture3D*>(pTextureResource) );
		break;
	default:
		LEAN_THROW_ERROR_MSG("Invalid texture resource type!");
	}

	return lean::bind_resource(pTexture);
}

} // namespace

// Loads a texture from the given file.
lean::resource_ptr<beGraphics::Texture, true> LoadTexture(const beGraphics::Device &device, const lean::utf8_ntri &fileName, const TextureDesc *pDesc, bool bSRGB)
{
	return DX11::CreateTexture( DX11::LoadTexture(ToImpl(device), fileName, pDesc, bSRGB).get() );
}

// Loads a texture from the given memory.
lean::resource_ptr<beGraphics::Texture, true> LoadTexture(const beGraphics::Device &device, const char *data, uint4 dataLength, const TextureDesc *pDesc, bool bSRGB)
{
	return DX11::CreateTexture( DX11::LoadTexture(ToImpl(device), data, dataLength, pDesc, bSRGB).get() );
}

// Creates a texture view from the given texture.
lean::resource_ptr<beGraphics::TextureView, true> ViewTexture(const beGraphics::Texture &texture, const beGraphics::Device &device)
{
	return lean::bind_resource<TextureView>(
			new DX11::TextureView( ToImpl(texture).GetResource(), nullptr, ToImpl(device) )
		);
}

// Gets the back buffer.
lean::resource_ptr<Texture, true> GetBackBuffer(const SwapChain &swapChain, uint4 index)
{
	return lean::bind_resource( new DX11::Texture2D( DX11::GetBackBuffer(ToImpl(swapChain).Get(), index).get() ) );
}

namespace DX11
{

// Creates a texture from the given description.
lean::com_ptr<ID3D11Texture1D, true> CreateTexture(const D3D11_TEXTURE1D_DESC &desc, const D3D11_SUBRESOURCE_DATA *pInitialData, ID3D11Device *pDevice)
{
	LEAN_ASSERT(pDevice != nullptr);

	lean::com_ptr<ID3D11Texture1D> pTexture;

	BE_THROW_DX_ERROR_MSG(
		pDevice->CreateTexture1D(&desc, pInitialData, pTexture.rebind()),
		"ID3D11Device::CreateTexture1D()");

	return pTexture.transfer();
}

// Creates a texture from the given description.
lean::com_ptr<ID3D11Texture2D, true> CreateTexture(const D3D11_TEXTURE2D_DESC &desc, const D3D11_SUBRESOURCE_DATA *pInitialData, ID3D11Device *pDevice)
{
	LEAN_ASSERT(pDevice != nullptr);

	lean::com_ptr<ID3D11Texture2D> pTexture;

	BE_THROW_DX_ERROR_MSG(
		pDevice->CreateTexture2D(&desc, pInitialData, pTexture.rebind()),
		"ID3D11Device::CreateTexture2D()");

	return pTexture.transfer();
}

// Creates a texture from the given description.
lean::com_ptr<ID3D11Texture3D, true> CreateTexture(const D3D11_TEXTURE3D_DESC &desc, const D3D11_SUBRESOURCE_DATA *pInitialData, ID3D11Device *pDevice)
{
	LEAN_ASSERT(pDevice != nullptr);

	lean::com_ptr<ID3D11Texture3D> pTexture;

	BE_THROW_DX_ERROR_MSG(
		pDevice->CreateTexture3D(&desc, pInitialData, pTexture.rebind()),
		"ID3D11Device::CreateTexture3D()");

	return pTexture.transfer();
}

// Creates a 2D texture.
lean::com_ptr<ID3D11Texture2D, true> CreateTexture2D(ID3D11Device *device, uint4 bindFlags, DXGI_FORMAT format,
	uint4 width, uint4 height, uint4 elements, uint4 mipLevels, uint4 flags)
{
	D3D11_TEXTURE2D_DESC desc;
	desc.Width = width;
	desc.Height = height;
	desc.ArraySize = elements;
	desc.MipLevels = mipLevels;
	desc.Format = format;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = bindFlags;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = flags;
	
	return CreateTexture(desc, nullptr, device);
}

// Creates a 3D texture.
lean::com_ptr<ID3D11Texture3D, true> CreateTexture3D(ID3D11Device *device, uint4 bindFlags, DXGI_FORMAT format,
	uint4 width, uint4 height, uint4 depth, uint4 mipLevels, uint4 flags)
{
	D3D11_TEXTURE3D_DESC desc;
	desc.Width = width;
	desc.Height = height;
	desc.Depth = depth;
	desc.MipLevels = mipLevels;
	desc.Format = format;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = bindFlags;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = flags;
	
	return CreateTexture(desc, nullptr, device);
}

/// Creates a staging texture matching the given texture.
lean::com_ptr<ID3D11Texture2D, true> CreateStagingTexture(ID3D11Device *device, ID3D11Texture2D *texture, uint4 element, uint4 mipLevel, uint4 cpuAccess)
{
	D3D11_TEXTURE2D_DESC desc;
	texture->GetDesc(&desc);
	desc.Usage = D3D11_USAGE_STAGING;
	desc.BindFlags = 0;
	desc.CPUAccessFlags = cpuAccess;
	desc.MiscFlags &= D3D11_RESOURCE_MISC_TEXTURECUBE;

	if (element != -1)
		desc.ArraySize = 1;
	if (mipLevel != -1)
	{
		desc.MipLevels = 1;

		for (uint4 i = 0; i < mipLevel; ++i)
		{
			desc.Width = max(desc.Width / 2, 1U);
			desc.Height = max(desc.Height / 2, 1U);
		}
	}

	return CreateTexture(desc, nullptr, device);
}

// Creates a staging texture matching the given texture.
lean::com_ptr<ID3D11Texture3D, true> CreateStagingTexture(ID3D11Device *device, ID3D11Texture3D *texture, uint4 mipLevel, uint4 cpuAccess)
{
	D3D11_TEXTURE3D_DESC desc;
	texture->GetDesc(&desc);
	desc.Usage = D3D11_USAGE_STAGING;
	desc.BindFlags = 0;
	desc.CPUAccessFlags = cpuAccess;
	desc.MiscFlags = 0;

	if (mipLevel != -1)
	{
		desc.MipLevels = 1;

		for (uint4 i = 0; i < mipLevel; ++i)
		{
			desc.Width = max(desc.Width / 2, 1U);
			desc.Height = max(desc.Height / 2, 1U);
			desc.Depth = max(desc.Depth / 2, 1U);
		}
	}

	return CreateTexture(desc, nullptr, device);
}

// Copies data from the given source texture to the given destination texture.
void CopyTexture(ID3D11DeviceContext *context, ID3D11Resource *dest, uint4 destOffset, ID3D11Resource *src, uint4 srcOffset)
{
	context->CopySubresourceRegion(dest, destOffset, 0, 0, 0, src, srcOffset, nullptr);
}

// Creates an unordered access view.
lean::com_ptr<ID3D11UnorderedAccessView, true> CreateUAV(ID3D11Resource *texture, const D3D11_UNORDERED_ACCESS_VIEW_DESC *pDesc)
{
	lean::com_ptr<ID3D11UnorderedAccessView> view;

	BE_THROW_DX_ERROR_MSG(
		GetDevice(*texture)->CreateUnorderedAccessView(texture, pDesc, view.rebind()),
		"ID3D11Device::CreateUnorderedAccessView()");

	return view.transfer();
}

// Creates a shader resource view.
lean::com_ptr<ID3D11ShaderResourceView, true> CreateSRV(ID3D11Resource *texture, const D3D11_SHADER_RESOURCE_VIEW_DESC *pDesc)
{
	lean::com_ptr<ID3D11ShaderResourceView> view;

	BE_THROW_DX_ERROR_MSG(
		GetDevice(*texture)->CreateShaderResourceView(texture, pDesc, view.rebind()),
		"ID3D11Device::CreateShaderResourceView()");

	return view.transfer();
}

/*
// Creates an unordered access view.
lean::com_ptr<ID3D11UnorderedAccessView, true> CreateMipUAV(ID3D11Resource *texture, uint4 mipLevel, DXGI_FORMAT format)
{
	GetDesc(

	D3D11_RESOURCE_DIMENSION dim;
	texture->GetType(&dim);

	switch (dim)
	{
	case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
		return GetDesc( static_cast<ID3D11Texture1D*>(pTexture) );
	case TextureType::Texture2D:
		return GetDesc( static_cast<ID3D11Texture2D*>(pTexture) );
	case TextureType::Texture3D:
		return GetDesc( static_cast<ID3D11Texture3D*>(pTexture) );
	}
}

// Creates a shader resource view.
lean::com_ptr<ID3D11ShaderResourceView, true> CreateMipSRV(ID3D11Resource *texture, uint4 mipLevel, DXGI_FORMAT format)
{
}
*/

// Creates a shader resource view from the given texture.
lean::com_ptr<ID3D11ShaderResourceView, true> CreateShaderResourceView(ID3D11Resource *texture, const D3D11_SHADER_RESOURCE_VIEW_DESC *pDesc, ID3D11Device *device)
{
	LEAN_ASSERT(texture != nullptr);
	LEAN_ASSERT(device != nullptr);

	lean::com_ptr<ID3D11ShaderResourceView> view;

	BE_THROW_DX_ERROR_MSG(
		device->CreateShaderResourceView(texture, pDesc, view.rebind()),
		"ID3D11Device::CreateShaderResourceView()");

	return view.transfer();
}

// Constructor.
template <TextureType::T Type>
TypedTexture<Type>::TypedTexture(const DescType &desc, const D3D11_SUBRESOURCE_DATA *pInitialData, ID3D11Device *pDevice)
	: m_pTexture( CreateTexture(desc, pInitialData, pDevice) )
{
}

// Constructor.
template <TextureType::T Type>
TypedTexture<Type>::TypedTexture(InterfaceType *pTexture)
	: m_pTexture(pTexture)
{
	LEAN_ASSERT(m_pTexture != nullptr);
}

// Destructor.
template <TextureType::T Type>
TypedTexture<Type>::~TypedTexture()
{
}

// Constructor.
TextureView::TextureView(ID3D11Resource *pTexture, const D3D11_SHADER_RESOURCE_VIEW_DESC *pDesc, ID3D11Device *pDevice)
	: m_pTexture( CreateShaderResourceView(pTexture, pDesc, pDevice) )
{
}

// Constructor.
TextureView::TextureView(ID3D11ShaderResourceView *pView)
	: m_pTexture( LEAN_ASSERT_NOT_NULL(pView) )
{
}

// Destructor.
TextureView::~TextureView()
{
}

// Maps this texture to allow for CPU access.
bool Map(ID3D11DeviceContext *pDeviceContext, ID3D11Resource *pTexture, uint4 subResource,
		D3D11_MAPPED_SUBRESOURCE &data, D3D11_MAP map, uint4 flags)
{
	LEAN_ASSERT(pDeviceContext != nullptr);
	LEAN_ASSERT(pTexture != nullptr);

	bool bSuccess = BE_LOG_DX_ERROR_MSG(
		pDeviceContext->Map(
			pTexture, subResource,
			map, flags,
			&data),
		"ID3D11DeviceContext::Map()");

	if (!bSuccess)
		data.pData = nullptr;

	return bSuccess;
}

// Unmaps this texture to allow for GPU access.
void Unmap(ID3D11DeviceContext *pDeviceContext, ID3D11Resource *pTexture, uint4 subResource)
{
	LEAN_ASSERT(pDeviceContext != nullptr);
	LEAN_ASSERT(pTexture != nullptr);

	pDeviceContext->Unmap(pTexture, subResource);
}

// Gets a description of the given texture.
D3D11_TEXTURE1D_DESC GetDesc(ID3D11Texture1D *pTexture)
{
	D3D11_TEXTURE1D_DESC desc;
	pTexture->GetDesc(&desc);
	return desc;
}

// Gets a description of the given texture.
D3D11_TEXTURE2D_DESC GetDesc(ID3D11Texture2D *pTexture)
{
	D3D11_TEXTURE2D_DESC desc;
	pTexture->GetDesc(&desc);
	return desc;
}

// Gets a description of the given texture.
D3D11_TEXTURE3D_DESC GetDesc(ID3D11Texture3D *pTexture)
{
	D3D11_TEXTURE3D_DESC desc;
	pTexture->GetDesc(&desc);
	return desc;
}

// Gets a description of the given texture.
TextureDesc GetTextureDesc(ID3D11Texture1D *pTexture)
{
	D3D11_TEXTURE1D_DESC desc;
	pTexture->GetDesc(&desc);
	return TextureDesc(desc.Width, 1, 1, FromAPI(desc.Format), desc.MipLevels);
}

// Gets a description of the given texture.
TextureDesc GetTextureDesc(ID3D11Texture2D *pTexture)
{
	D3D11_TEXTURE2D_DESC desc;
	pTexture->GetDesc(&desc);
	return TextureDesc(desc.Width, desc.Height, 1, FromAPI(desc.Format), desc.MipLevels);
}

// Gets a description of the given texture.
TextureDesc GetTextureDesc(ID3D11Texture3D *pTexture)
{
	D3D11_TEXTURE3D_DESC desc;
	pTexture->GetDesc(&desc);
	return TextureDesc(desc.Width, desc.Height, desc.Depth, FromAPI(desc.Format), desc.MipLevels);
}

// Gets a description of the given texture.
TextureDesc GetTextureDesc(ID3D11Resource *pTexture)
{
	switch ( GetTextureType(pTexture) )
	{
	case TextureType::Texture1D:
		return GetTextureDesc( static_cast<ID3D11Texture1D*>(pTexture) );
	case TextureType::Texture2D:
		return GetTextureDesc( static_cast<ID3D11Texture2D*>(pTexture) );
	case TextureType::Texture3D:
		return GetTextureDesc( static_cast<ID3D11Texture3D*>(pTexture) );
	}

	return TextureDesc();
}

// Gets the type of the given texture.
TextureType::T GetTextureType(ID3D11Resource *pTexture)
{
	D3D11_RESOURCE_DIMENSION resourceDim;
	pTexture->GetType(&resourceDim);

	switch (resourceDim)
	{
	case D3D11_RESOURCE_DIMENSION_TEXTURE1D: return TextureType::Texture1D;
	case D3D11_RESOURCE_DIMENSION_TEXTURE2D: return TextureType::Texture2D;
	case D3D11_RESOURCE_DIMENSION_TEXTURE3D: return TextureType::Texture3D;
	}

	return TextureType::NotATexture;
}

// Gets data from the given texture.
bool ReadTextureData(ID3D11DeviceContext *context, ID3D11Resource *texture,
	void *bytes, uint4 rowByteCount, uint4 rowCount, uint4 sliceCount, uint4 subResource)
{
	D3D11_MAPPED_SUBRESOURCE mapped;

	if (BE_LOG_DX_ERROR_MSG(
			context->Map(
					texture, subResource,
					D3D11_MAP_READ, 0,
					&mapped
				),
			"ID3D11DeviceContext::Map()" )
		)
	{
		char *dest = static_cast<char*>(bytes);
		
		for (uint4 j = 0; j < sliceCount; ++j)
		{
			const char *src = static_cast<const char*>(mapped.pData) + (j * mapped.DepthPitch);

			for (uint4 i = 0; i < rowCount; ++i)
			{
				memcpy(dest, src, rowByteCount);
				dest += rowByteCount;
				src += mapped.RowPitch;
			}
		}

		context->Unmap(texture, subResource);

		return true;
	}
	else
		return false;
}

// Gets data from the given texture using a TEMPORARY texture texture. SLOW!
template <class ID3D11TextureInterface>
LEAN_INLINE bool DebugFetchTextureData(ID3D11DeviceContext *context, ID3D11TextureInterface *texture,
	void *bytes, uint4 rowByteCount, uint4 rowCount, uint4 sliceCount, uint4 subResource)
{
	bool result;

	try
	{
		lean::com_ptr<ID3D11TextureInterface> stagingTexture = CreateStagingTexture(GetDevice(*texture), texture);

		context->CopyResource(stagingTexture, texture);

		result = ReadTextureData(context, stagingTexture, bytes, rowByteCount, rowCount, sliceCount, subResource);
	}
	catch (const std::runtime_error &)
	{
		result = false;
	}

	return result;
}

// Gets data from the given texture using a TEMPORARY texture texture. SLOW!
bool DebugFetchTextureData(ID3D11DeviceContext *context, ID3D11Texture2D *texture,
	void *bytes, uint4 rowByteCount, uint4 rowCount, uint4 subResource)
{
	return DebugFetchTextureData<ID3D11Texture2D>(context, texture, bytes, rowByteCount, rowCount, 1, subResource);
}

// Gets data from the given texture using a TEMPORARY texture texture. SLOW!
bool DebugFetchTextureData(ID3D11DeviceContext *context, ID3D11Texture3D *texture,
	void *bytes, uint4 rowByteCount, uint4 rowCount, uint4 sliceCount, uint4 subResource)
{
	return DebugFetchTextureData<ID3D11Texture3D>(context, texture, bytes, rowByteCount, rowCount, sliceCount, subResource);
}

} // namespace

} // namespace
