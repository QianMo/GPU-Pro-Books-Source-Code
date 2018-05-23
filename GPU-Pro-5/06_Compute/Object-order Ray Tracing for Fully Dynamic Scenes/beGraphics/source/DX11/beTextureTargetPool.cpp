/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#include "beGraphicsInternal/stdafx.h"

#include "beGraphics/DX11/beTextureTargetPool.h"
#include "beGraphics/DX11/beTexture.h"
#include "beGraphics/DX11/beFormat.h"
#include "beGraphics/DX11/beDeviceContext.h"
#include "beGraphics/DX/beError.h"

#include <boost/ptr_container/ptr_vector.hpp>

#include <lean/functional/algorithm.h>

#include <lean/logging/log.h>

namespace beGraphics
{

namespace DX11
{

namespace
{

/// Compares the given pair of descriptions.
LEAN_INLINE bool DescLessThan(const TextureTargetDesc &left, const TextureTargetDesc &right)
{
	return (memcmp(&left, &right, sizeof(TextureTargetDesc)) < 0);
}

/// Orders texture target pointers by the descriptions of the texture targets pointed to.
struct TargetPointerDescCompare
{
	LEAN_INLINE bool operator ()(const TextureTarget *left, const TextureTarget &right)
	{
		return DescLessThan(left->GetDesc(), right.GetDesc());
	}
	LEAN_INLINE bool operator ()(const TextureTarget &left, const TextureTarget *right)
	{
		return DescLessThan(left.GetDesc(), right->GetDesc());
	}
	LEAN_INLINE bool operator ()(const TextureTarget &left, const TextureTarget &right)
	{
		return DescLessThan(left.GetDesc(), right.GetDesc());
	}
	LEAN_INLINE bool operator ()(const TextureTarget &left, const TextureTargetDesc &right)
	{
		return DescLessThan(left.GetDesc(), right);
	}
	LEAN_INLINE bool operator ()(const TextureTargetDesc &left, const TextureTarget &right)
	{
		return DescLessThan(left, right.GetDesc());
	}
};

/// Converts the given texture target description into a DX11 texture description.
D3D11_TEXTURE2D_DESC ToAPI(const TextureTargetDesc &desc, UINT bindFlags, DXGI_FORMAT formatOverride = DXGI_FORMAT_UNKNOWN)
{
	D3D11_TEXTURE2D_DESC descDX;
	descDX.Width = desc.Width;
	descDX.Height = desc.Height;
	descDX.MipLevels = desc.MipLevels & ~TextureTargetFlags::AutoGenMipMaps;
	descDX.ArraySize = desc.Count;
	descDX.Format = (formatOverride != DXGI_FORMAT_UNKNOWN) ? formatOverride : desc.Format;
	descDX.SampleDesc = desc.Samples;
	descDX.Usage = D3D11_USAGE_DEFAULT;
	descDX.BindFlags = bindFlags;
	descDX.CPUAccessFlags = 0;
	descDX.MiscFlags = (desc.MipLevels & TextureTargetFlags::AutoGenMipMaps) ? D3D11_RESOURCE_MISC_GENERATE_MIPS : 0;
	return descDX;
}

} // namespace

} // namespace

// Constructor. Texture ist OPTIONAL.
TextureTarget::TextureTarget(const DX11::TextureTargetDesc &desc, ID3D11Resource *pResource, ID3D11ShaderResourceView *pTexture, texture_ptr *pTextures)
	: m_desc(desc),
	m_pResource(pResource),
	m_pTexture(pTexture),
	m_pTextures(pTextures),
	m_references(0)
{
}

// Destructor.
TextureTarget::~TextureTarget()
{
}

// Constructor. Texture ist OPTIONAL.
ColorTextureTarget::ColorTextureTarget(const DX11::TextureTargetDesc &desc, ID3D11Resource *pResource,
	ID3D11ShaderResourceView *pTexture, ID3D11RenderTargetView *pTarget, target_ptr *pTargets, texture_ptr *pTextures)
	: TextureTarget(desc, pResource, pTexture, pTextures),
	m_pTarget( LEAN_ASSERT_NOT_NULL(pTarget) ),
	// TODO: Public, might transfer cross-module memory!
	// MAY be nullptr
	m_pTargets( pTargets )
{
}

// Destructor.
ColorTextureTarget::~ColorTextureTarget()
{
}

// Constructor. Texture ist OPTIONAL.
DepthStencilTextureTarget::DepthStencilTextureTarget(const DX11::TextureTargetDesc &desc, ID3D11Resource *pResource,
	ID3D11ShaderResourceView *pTexture, ID3D11DepthStencilView *pTarget, target_ptr *pTargets)
	: TextureTarget(desc, pResource, pTexture),
	m_pTarget( LEAN_ASSERT_NOT_NULL(pTarget) ),
	// TODO: Public, might transfer cross-module memory!
	// MAY be nullptr
	m_pTargets( pTargets )
{
}

// Destructor.
DepthStencilTextureTarget::~DepthStencilTextureTarget()
{
}

// Constructor. Texture ist OPTIONAL.
StageTextureTarget::StageTextureTarget(const DX11::TextureTargetDesc &desc, ID3D11Resource *pResource)
	: TextureTarget(desc, pResource, nullptr)
{
}

// Destructor.
StageTextureTarget::~StageTextureTarget()
{
}

namespace DX11
{

namespace
{

/// Returns a free texture target in the given sorted range, if available.
template <class TextureTarget, class Iterator>
TextureTarget* FindFreeTextureTarget(Iterator begin, Iterator end, const TextureTargetDesc &desc)
{
	std::pair<Iterator, Iterator> targetRange = std::equal_range( begin, end, desc, TargetPointerDescCompare() );

	for (Iterator itTarget = targetRange.first; itTarget != targetRange.second; ++itTarget)
	{
		TextureTarget &target = *itTarget;

		if (!target.IsInUse())
			return &target;
	}

	return nullptr;
}

/// Resets usage.
template <class Iterator>
void ResetTargetUsage(Iterator begin, Iterator end)
{
	for (Iterator it = begin; it != end; ++it)
		it->ResetUses();
}

/// Resets usage.
template <class Vector>
uint4 RemoveUnusedTargets(Vector &targets)
{
	uint4 count = 0;

	for (typename Vector::iterator it = targets.begin(); it != targets.end(); )
		if (!it->WasUsed())
		{
			it = targets.erase(it);
			++count;
		}
		else
			++it;

	return count;
}

} // namespace

struct TextureTargetPool::M
{
	lean::com_ptr<ID3D11Device> pDevice;

	typedef boost::ptr_vector<ColorTextureTarget> color_target_vector;
	color_target_vector colorTargets;
	typedef boost::ptr_vector<DepthStencilTextureTarget> depth_stencil_target_vector;
	depth_stencil_target_vector depthStencilTargets;
	typedef boost::ptr_vector<StageTextureTarget> stage_target_vector;
	stage_target_vector stageTargets;

	M(ID3D11Device *pDevice)
		: pDevice( LEAN_ASSERT_NOT_NULL(pDevice) )
	{
	};
};

// Constructor.
TextureTargetPool::TextureTargetPool(ID3D11Device *pDevice)
	: m( new M(pDevice) )
{
}

// Destructor.
TextureTargetPool::~TextureTargetPool()
{
}

// Resets the usage statistics.
void TextureTargetPool::ResetUsage()
{
	ResetTargetUsage(m->colorTargets.begin(), m->colorTargets.end());
	ResetTargetUsage(m->depthStencilTargets.begin(), m->depthStencilTargets.end());
	ResetTargetUsage(m->stageTargets.begin(), m->stageTargets.end());
}

// Releases unused targets.
void TextureTargetPool::ReleaseUnused()
{
	uint4 releaseCount = 0;
	
	releaseCount += RemoveUnusedTargets(m->colorTargets);
	releaseCount += RemoveUnusedTargets(m->depthStencilTargets);
	releaseCount += RemoveUnusedTargets(m->stageTargets);

	if (releaseCount > 0)
		LEAN_LOG("TextureTargetPool released " << releaseCount << " intermediate targets.");
}

// Acquires a color texture target according to the given description.
lean::com_ptr<const ColorTextureTarget, true> TextureTargetPool::AcquireColorTextureTarget(const TextureTargetDesc &desc)
{
	ColorTextureTarget *pTarget = FindFreeTextureTarget<ColorTextureTarget>(
		m->colorTargets.begin(), m->colorTargets.end(), desc);

	if (!pTarget)
	{
		lean::com_ptr<ID3D11Texture2D> pTexture;
		lean::com_ptr<ID3D11ShaderResourceView> pTextureView;

		D3D11_TEXTURE2D_DESC descDX = ToAPI(desc, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE);

		try
		{
			pTexture = CreateTexture(descDX, nullptr, m->pDevice);

			BE_LOG_DX_ERROR_MSG(
				m->pDevice->CreateShaderResourceView(pTexture, nullptr, pTextureView.rebind()),
				"ID3D11Device::CreateShaderResourceView()");
		}
		catch (const std::runtime_error&)
		{
			// Retry write-only
			descDX.BindFlags = D3D11_BIND_RENDER_TARGET;
			descDX.MiscFlags = 0;

			pTexture = CreateTexture(descDX, nullptr, m->pDevice);
		}

		lean::com_ptr<ID3D11RenderTargetView> pTargetView;
		ColorTextureTarget::scoped_target_array_ptr pTargetViews;
		ColorTextureTarget::scoped_texture_array_ptr pTextureViews;

		// Create element views
		if (descDX.ArraySize > 1 || descDX.MipLevels != 1)
		{
			// NOTE: Get actual number of mip levels
			pTexture->GetDesc(&descDX);

			uint4 trueMipLevels = !(descDX.MiscFlags & D3D11_RESOURCE_MISC_GENERATE_MIPS) ? descDX.MipLevels : 1;

			pTargetViews = new ColorTextureTarget::target_ptr[descDX.ArraySize * trueMipLevels];
			if (pTextureView)
				pTextureViews = new ColorTextureTarget::texture_ptr[descDX.ArraySize * trueMipLevels];

			D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
			rtvDesc.Format = descDX.Format;

			D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
			srvDesc.Format = descDX.Format;

			for (uint4 j = 0; j < trueMipLevels; ++j)
				for (uint4 i = 0; i < descDX.ArraySize; ++i)
				{
					if (descDX.SampleDesc.Count > 1)
					{
						rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY;
						rtvDesc.Texture2DMSArray.FirstArraySlice = i;
						rtvDesc.Texture2DMSArray.ArraySize = 1;

						srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY;
						srvDesc.Texture2DMSArray.FirstArraySlice = i;
						srvDesc.Texture2DMSArray.ArraySize = 1;
					}
					else
					{
						rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
						rtvDesc.Texture2DArray.MipSlice = j;
						rtvDesc.Texture2DArray.FirstArraySlice = i;
						rtvDesc.Texture2DArray.ArraySize = 1;

						srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
						srvDesc.Texture2DArray.MostDetailedMip = j;
						srvDesc.Texture2DArray.MipLevels = 1;
						srvDesc.Texture2DArray.FirstArraySlice = i;
						srvDesc.Texture2DArray.ArraySize = 1;
					}

					BE_THROW_DX_ERROR_MSG(
						m->pDevice->CreateRenderTargetView(pTexture, &rtvDesc, pTargetViews[j * descDX.ArraySize + i].rebind()),
						"ID3D11Device::CreateRenderTargetView()");

					if (pTextureViews)
						BE_THROW_DX_ERROR_MSG(
							m->pDevice->CreateShaderResourceView(pTexture, &srvDesc, pTextureViews[j * descDX.ArraySize + i].rebind()),
							"ID3D11Device::CreateShaderResourceView()");
				}
			}

		// Create array view
		BE_THROW_DX_ERROR_MSG(
			m->pDevice->CreateRenderTargetView(pTexture, nullptr, pTargetView.rebind()),
			"ID3D11Device::CreateRenderTargetView()");

		pTarget = new ColorTextureTarget(
				desc, pTexture,
				pTextureView, pTargetView,
				pTargetViews.detach(),
				pTextureViews.detach()
			);
		lean::push_sorted(m->colorTargets, pTarget, TargetPointerDescCompare());
	}

	return lean::com_ptr<const ColorTextureTarget, true>(pTarget);
}

// Acquires a depth-stencil texture target according to the given description.
lean::com_ptr<const DepthStencilTextureTarget, true> TextureTargetPool::AcquireDepthStencilTextureTarget(const TextureTargetDesc &desc)
{
	DepthStencilTextureTarget *pTarget = FindFreeTextureTarget<DepthStencilTextureTarget>(
		m->depthStencilTargets.begin(), m->depthStencilTargets.end(), desc);

	if (!pTarget)
	{
		DXGI_FORMAT resFormat = desc.Format, srvFormat = desc.Format, dsvFormat = desc.Format;

		// Depth-stencil resource format fix-up
		switch (SizeofFormat(desc.Format))
		{
		case 2:
			if (desc.Format != DXGI_FORMAT_D16_UNORM)
			{
				resFormat = DXGI_FORMAT_R16_TYPELESS;
				dsvFormat = DXGI_FORMAT_D16_UNORM;
			}
			break;

		case 4:
			if (desc.Format == DXGI_FORMAT_R24_UNORM_X8_TYPELESS || desc.Format == DXGI_FORMAT_X24_TYPELESS_G8_UINT)
			{
				resFormat = DXGI_FORMAT_R24G8_TYPELESS;
				dsvFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
			}
			else if (desc.Format != DXGI_FORMAT_D32_FLOAT)
			{
				resFormat = DXGI_FORMAT_R32_TYPELESS;
				dsvFormat = DXGI_FORMAT_D32_FLOAT;
			}
			break;
		}

		lean::com_ptr<ID3D11Texture2D> pTexture;
		lean::com_ptr<ID3D11ShaderResourceView> pTextureView;

		D3D11_TEXTURE2D_DESC descDX = ToAPI(desc, D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE, resFormat);
		// Unsupported for DS!
		descDX.MipLevels = 1;
		descDX.MiscFlags = 0;

		try
		{
			pTexture = CreateTexture(descDX, nullptr, m->pDevice);

			// Get actual mip level count AFTER creation
			pTexture->GetDesc(&descDX);

			D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
			srvDesc.Format = srvFormat;
			
			if (descDX.ArraySize > 1)
			{
				if (descDX.SampleDesc.Count > 1)
				{
					srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY;
					srvDesc.Texture2DMSArray.FirstArraySlice = 0;
					srvDesc.Texture2DMSArray.ArraySize = descDX.ArraySize;
				}
				else
				{
					srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
					srvDesc.Texture2DArray.MostDetailedMip = 0;
					srvDesc.Texture2DArray.MipLevels = descDX.MipLevels;
					srvDesc.Texture2DArray.FirstArraySlice = 0;
					srvDesc.Texture2DArray.ArraySize = descDX.ArraySize;
				}
			}
			else
			{
				srvDesc.ViewDimension = (descDX.SampleDesc.Count > 1) ? D3D11_SRV_DIMENSION_TEXTURE2DMS : D3D11_SRV_DIMENSION_TEXTURE2D;
				srvDesc.Texture2D.MostDetailedMip = 0;
				srvDesc.Texture2D.MipLevels = descDX.MipLevels;
			}

			BE_LOG_DX_ERROR_MSG(
				m->pDevice->CreateShaderResourceView(pTexture, &srvDesc, pTextureView.rebind()),
				"ID3D11Device::CreateShaderResourceView()");
		}
		catch (const std::runtime_error&)
		{
			// Retry write-only
			descDX.BindFlags = D3D11_BIND_DEPTH_STENCIL;
			descDX.MiscFlags = 0;

			pTexture = CreateTexture(descDX, nullptr, m->pDevice);
		}

		lean::com_ptr<ID3D11DepthStencilView> pTargetView;
		DepthStencilTextureTarget::scoped_target_array_ptr pTargetViews;

		D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc;
		dsvDesc.Format = dsvFormat;
		dsvDesc.Flags = 0;

		if (descDX.ArraySize > 1)
		{
			// Create element views
			pTargetViews = new DepthStencilTextureTarget::target_ptr[descDX.ArraySize];

			for (uint4 i = 0; i < descDX.ArraySize; ++i)
			{
				if (descDX.SampleDesc.Count > 1)
				{
					dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY;
					dsvDesc.Texture2DMSArray.FirstArraySlice = i;
					dsvDesc.Texture2DMSArray.ArraySize = 1;
				}
				else
				{
					dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
					dsvDesc.Texture2DArray.MipSlice = 0;
					dsvDesc.Texture2DArray.FirstArraySlice = i;
					dsvDesc.Texture2DArray.ArraySize = 1;
				}

				BE_THROW_DX_ERROR_MSG(
					m->pDevice->CreateDepthStencilView(pTexture, &dsvDesc, pTargetViews[i].rebind()),
					"ID3D11Device::CreateDepthStencilView()");
			}

			// Create array view
			if (descDX.SampleDesc.Count > 1)
			{
				dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY;
				dsvDesc.Texture2DMSArray.FirstArraySlice = 0;
				dsvDesc.Texture2DMSArray.ArraySize = descDX.ArraySize;
			}
			else
			{
				dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
				dsvDesc.Texture2DArray.MipSlice = 0;
				dsvDesc.Texture2DArray.FirstArraySlice = 0;
				dsvDesc.Texture2DArray.ArraySize = descDX.ArraySize;
			}

			BE_THROW_DX_ERROR_MSG(
				m->pDevice->CreateDepthStencilView(pTexture, &dsvDesc, pTargetView.rebind()),
				"ID3D11Device::CreateDepthStencilView()");
		}
		else
		{
			dsvDesc.ViewDimension = (descDX.SampleDesc.Count > 1) ? D3D11_DSV_DIMENSION_TEXTURE2DMS : D3D11_DSV_DIMENSION_TEXTURE2D;
			dsvDesc.Texture2D.MipSlice = 0;

			BE_THROW_DX_ERROR_MSG(
				m->pDevice->CreateDepthStencilView(pTexture, &dsvDesc, pTargetView.rebind()),
				"ID3D11Device::CreateDepthStencilView()");
		}

		pTarget = new DepthStencilTextureTarget(
				desc, pTexture,
				pTextureView, pTargetView,
				pTargetViews.detach()
			);
		lean::push_sorted(m->depthStencilTargets, pTarget, TargetPointerDescCompare());
	}

	return lean::com_ptr<const DepthStencilTextureTarget, true>(pTarget);
}

// Acquires a stage texture target according to the given description.
lean::com_ptr<const StageTextureTarget, true> TextureTargetPool::AcquireStageTextureTarget(const TextureTargetDesc &desc)
{
	StageTextureTarget *pTarget = FindFreeTextureTarget<StageTextureTarget>(
		m->stageTargets.begin(), m->stageTargets.end(), desc);

	if (!pTarget)
	{
		lean::com_ptr<ID3D11Texture2D> pTexture;
		
		D3D11_TEXTURE2D_DESC descDX = ToAPI(desc, 0);
		descDX.Usage = D3D11_USAGE_STAGING;
		descDX.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;

		pTexture = CreateTexture(descDX, nullptr, m->pDevice);

		pTarget = new StageTextureTarget(desc, pTexture);
		lean::push_sorted(m->stageTargets, pTarget, TargetPointerDescCompare());
	}

	return lean::com_ptr<const StageTextureTarget, true>(pTarget);
}

// Schedules the given subresource for read back.
lean::com_ptr<const StageTextureTarget, true> TextureTargetPool::ScheduleReadback(const ColorTextureTarget *target, const beg::DeviceContext &context, uint4 subResource)
{
	ID3D11Texture2D *targetResource = static_cast<ID3D11Texture2D*>(target->GetResource());
	D3D11_TEXTURE2D_DESC targetDesc = GetDesc(targetResource);

	uint4 mipLevel = subResource % targetDesc.MipLevels;

	TextureTargetDesc stageDesc;
	stageDesc.Format = targetDesc.Format;
	stageDesc.Width = max(targetDesc.Width >> mipLevel, 1U);
	stageDesc.Height = max(targetDesc.Height >> mipLevel, 1U);
	stageDesc.Samples = targetDesc.SampleDesc;
	stageDesc.MipLevels = 1;
	stageDesc.Count = 1;

	lean::com_ptr<const StageTextureTarget> stageTarget = AcquireStageTextureTarget(stageDesc);
	ToImpl(context)->CopySubresourceRegion(stageTarget->GetResource(), 0, 0, 0, 0, targetResource, subResource, nullptr);

	return stageTarget.transfer();
}

// Reads back from the given stage target.
bool TextureTargetPool::ReadBack(const StageTextureTarget *target, void *memory, uint4 size, const beg::DeviceContext &context)
{
	ID3D11Texture2D *targetResource = static_cast<ID3D11Texture2D*>(target->GetResource());
	D3D11_TEXTURE2D_DESC stageDesc = GetDesc(targetResource);

	return ReadTextureData(ToImpl(context), targetResource, memory, size / stageDesc.Height, stageDesc.Height, 1, 0);
}

// Reads back color target information.
bool TextureTargetPool::ReadBack(const ColorTextureTarget *target, void *memory, uint4 size, const beGraphics::DeviceContext &context, uint4 subResource)
{
	return ReadBack(ScheduleReadback(target, context, subResource).get(), memory, size, context);
}

} // namespace

} // namespace
