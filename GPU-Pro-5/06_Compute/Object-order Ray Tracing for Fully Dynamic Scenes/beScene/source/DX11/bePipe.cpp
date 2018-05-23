/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/DX11/bePipe.h"

#include <lean/functional/algorithm.h>

#include <beGraphics/DX/beError.h>

namespace beScene
{

namespace DX11
{

/// Target.
struct Target
{
	utf8_string name;
	uint4 flags;
	PipeOutputMask used;

	Target(const utf8_ntri &name, uint4 flags = 0, PipeOutputMask used = 0)
		: name(name.to<utf8_string>()),
		flags(flags),
		used(used) { }
};

struct Pipe::ColorTarget : public Target
{
	lean::com_ptr<const beGraphics::Any::ColorTextureTarget> pTarget;

	ColorTarget(const utf8_ntri &name, uint4 flags = 0,
		const beGraphics::Any::ColorTextureTarget *pTarget = nullptr)
			: Target(name, flags),
			pTarget(pTarget) { }
};

struct Pipe::DepthStencilTarget : public Target
{
	lean::com_ptr<const beGraphics::Any::DepthStencilTextureTarget> pTarget;

	DepthStencilTarget(const utf8_ntri &name, uint4 flags = 0,
		const beGraphics::Any::DepthStencilTextureTarget *pTarget = nullptr)
			: Target(name, flags),
			pTarget(pTarget) { }
};

namespace
{

/// Sorts texture targets by their names.
struct TargetNameCompare
{
	LEAN_INLINE bool operator ()(const Target &left, const Target &right) { return left.name < right.name; }
	LEAN_INLINE bool operator ()(const Target &left, const utf8_ntri &right) { return left.name < right; }
	LEAN_INLINE bool operator ()(const utf8_ntri &left, const Target &right) { return left < right.name; }
};

/// Gets the target identified by the given name.
template <class TargetVector>
inline typename TargetVector::iterator GetTarget(TargetVector &targets, const utf8_ntri &name)
{
	typename TargetVector::iterator it = std::lower_bound(targets.begin(), targets.end(), name, TargetNameCompare());
	return (it != targets.end() && it->name == name)
		? it
		: targets.end();
}
/// Gets the target identified by the given name.
template <class TargetVector>
inline typename TargetVector::const_iterator GetTarget(const TargetVector &targets, const utf8_ntri &name)
{
	typename TargetVector::const_iterator it = std::lower_bound(targets.begin(), targets.end(), name, TargetNameCompare());
	return (it != targets.end() && it->name == name)
		? it
		: targets.end();
}

/// Adds a new target to the given color target vector.
template <class Target, class TargetVector>
inline typename TargetVector::iterator AddTarget(TargetVector &targets, const utf8_ntri &name)
{
	targets.push_back( Target(name) );
	return lean::insert_last(targets.begin(), --targets.end(), TargetNameCompare());
}

/// Gets the target identified by the given name or adds a new target if none available.
template <class Target, class TargetVector>
inline typename TargetVector::iterator GetOrAddTarget(TargetVector &targets, const utf8_ntri &name)
{
	typename TargetVector::iterator itTarget = GetTarget(targets, name);
	return (itTarget != targets.end())
		? itTarget
		: AddTarget<Target>(targets, name);
}

/// Gets a texture target description template from the given texture.
beGraphics::Any::TextureTargetDesc GetTargetDescTemplate(ID3D11Texture2D *pTexture)
{
	beGraphics::Any::TextureTargetDesc desc;
	
	D3D11_TEXTURE2D_DESC descDX;
	pTexture->GetDesc(&descDX);

	desc.Width = descDX.Width;
	desc.Height = descDX.Height;
	desc.MipLevels = 1;
	desc.Format = descDX.Format;
	desc.Samples = descDX.SampleDesc;
	desc.Count = 1;
	return desc;
}

/// Gets a texture target description template from the given texture.
beGraphics::Any::TextureTargetDesc CompleteTargetDesc(const beGraphics::Any::TextureTargetDesc &descProto, const beGraphics::Any::TextureTargetDesc &descTemplate, uint4 flags)
{
	beGraphics::Any::TextureTargetDesc desc(descProto);
	
	if (desc.Width == 0)
		desc.Width = descTemplate.Width;
	if (desc.Height == 0)
		desc.Height = descTemplate.Height;
	if (desc.Format == DXGI_FORMAT_UNKNOWN)
		desc.Format = descTemplate.Format;
	if (desc.Samples.Count == 0)
		desc.Samples = descTemplate.Samples;
	if (flags & PipeTargetFlags::Output)
		desc.Count = descTemplate.Count;

	return desc;
}

} // namespace

// Constructor.
Pipe::Pipe(const beGraphics::Any::Texture &finalTarget, beGraphics::Any::TextureTargetPool *pTargetPool)
	: m_desc( new beGraphics::Any::TextureTargetDesc(GetTargetDescTemplate( beGraphics::Any::ToTex<beGraphics::TextureType::Texture2D>(finalTarget) )) ),
	m_pTargetPool( LEAN_ASSERT_NOT_NULL(pTargetPool) ),
	m_bKeepResults( false )
{
	LEAN_ASSERT(m_desc->Count <= MaxPipeOutputCount);

	SetFinalTarget(&finalTarget);
}

// Constructor.
Pipe::Pipe(const beGraphics::Any::TextureTargetDesc &desc, beGraphics::Any::TextureTargetPool *pTargetPool)
	: m_desc( new beGraphics::Any::TextureTargetDesc(desc) ),
	m_pTargetPool( LEAN_ASSERT_NOT_NULL(pTargetPool) ),
	m_bKeepResults( false )
{
	LEAN_ASSERT(m_desc->Count <= MaxPipeOutputCount);
}

// Destructor.
Pipe::~Pipe()
{
}

// Gets a new color target matching the given description.
lean::com_ptr<const beGraphics::Any::ColorTextureTarget, true> Pipe::NewColorTarget(const beGraphics::Any::TextureTargetDesc &desc, uint4 flags) const
{
	return m_pTargetPool->AcquireColorTextureTarget( CompleteTargetDesc(desc, *m_desc, flags) );
}

// Gets a new depth-stencil target matching the given description.
lean::com_ptr<const beGraphics::Any::DepthStencilTextureTarget, true> Pipe::NewDepthStencilTarget(const beGraphics::Any::TextureTargetDesc &desc, uint4 flags) const
{
	return m_pTargetPool->AcquireDepthStencilTextureTarget( CompleteTargetDesc(desc, *m_desc, flags) );
}

// Gets the target identified by the given name or nullptr if none available.
const beGraphics::Any::TextureTarget* Pipe::GetAnyTarget(const utf8_ntri &name) const
{
	const beGraphics::TextureTarget* pTarget = GetColorTarget(name);
	return (pTarget)
		? pTarget
		: GetDepthStencilTarget(name);
}

// Gets the color target identified by the given name or nullptr if none available.
const beGraphics::Any::ColorTextureTarget* Pipe::GetColorTarget(const utf8_ntri &name) const
{
	color_target_vector::const_iterator itTarget = GetTarget(m_colorTargets, name);
	return (itTarget != m_colorTargets.end())
		? itTarget->pTarget
		: nullptr;
}

// Gets the depth-stencil target identified by the given name nullptr if none available.
const beGraphics::Any::DepthStencilTextureTarget* Pipe::GetDepthStencilTarget(const utf8_ntri &name) const
{
	depth_stencil_target_vector::const_iterator itTarget = GetTarget(m_depthStencilTargets, name);
	return (itTarget != m_depthStencilTargets.end())
		? itTarget->pTarget
		: nullptr;
}

// Updates the color target identified by the given name.
void Pipe::SetColorTarget(const utf8_ntri &name, const beGraphics::Any::ColorTextureTarget *pTarget, uint4 flags, uint4 outputIndex, 
	lean::com_ptr<const beGraphics::Any::ColorTextureTarget> *pOldTarget)
{
	ColorTarget &target = *GetOrAddTarget<ColorTarget>(m_colorTargets, name);
	
	if (pOldTarget)
		*pOldTarget = target.pTarget;

	if (~target.flags & PipeTargetFlags::Immutable | flags & PipeTargetFlags::Flash)
	{
		bool bWasValid = (target.pTarget != nullptr);
		target.pTarget = pTarget;
		target.used = (1 << outputIndex);

		if (!bWasValid | ~flags & PipeTargetFlags::Keep)
			target.flags = flags;
	}
}

// Updates the color target identified by the given name.
void Pipe::SetDepthStencilTarget(const utf8_ntri &name, const beGraphics::Any::DepthStencilTextureTarget *pTarget, uint4 flags, uint4 outputIndex, 
	lean::com_ptr<const beGraphics::Any::DepthStencilTextureTarget> *pOldTarget)
{
	DepthStencilTarget &target = *GetOrAddTarget<DepthStencilTarget>(m_depthStencilTargets, name);
	
	if (pOldTarget)
		*pOldTarget = target.pTarget;

	if (~target.flags & PipeTargetFlags::Immutable | flags & PipeTargetFlags::Flash)
	{
		bool bWasValid = (target.pTarget != nullptr);
		target.pTarget = pTarget;
		target.used = (1 << outputIndex);

		if (!bWasValid | ~flags & PipeTargetFlags::Keep)
			target.flags = flags;
	}
}

// Gets a new color target matching the given description and stores it under the given name.
const beGraphics::Any::ColorTextureTarget* Pipe::GetNewColorTarget(const utf8_ntri &name, const beGraphics::Any::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex, 
	lean::com_ptr<const beGraphics::Any::ColorTextureTarget> *pOldTarget)
{
	ColorTarget &target = *GetOrAddTarget<ColorTarget>(m_colorTargets, name);

	if (pOldTarget)
		*pOldTarget = target.pTarget;

	if (~target.flags & PipeTargetFlags::Immutable | flags & PipeTargetFlags::Flash)
	{
		bool bWasValid = (target.pTarget != nullptr);
		target.pTarget = NewColorTarget(desc, flags);
		target.used = (1 << outputIndex);
		
		if (!bWasValid | ~flags & PipeTargetFlags::Keep)
			target.flags = flags;
	}
	else
		target.used |= (1 << outputIndex);

	return target.pTarget;
}

// Gets a new depth-stencil target matching the given description and stores it under the given name.
const beGraphics::Any::DepthStencilTextureTarget* Pipe::GetNewDepthStencilTarget(const utf8_ntri &name, const beGraphics::Any::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex, 
	lean::com_ptr<const beGraphics::Any::DepthStencilTextureTarget> *pOldTarget)
{
	DepthStencilTarget &target = *GetOrAddTarget<DepthStencilTarget>(m_depthStencilTargets, name);

	if (pOldTarget)
		*pOldTarget = target.pTarget;

	if (~target.flags & PipeTargetFlags::Immutable | flags & PipeTargetFlags::Flash)
	{
		bool bWasValid = (target.pTarget != nullptr);
		target.pTarget = NewDepthStencilTarget(desc, flags);
		target.used = (1 << outputIndex);

		if (!bWasValid | ~flags & PipeTargetFlags::Keep)
			target.flags = flags;
	}
	else
		target.used |= (1 << outputIndex);

	return target.pTarget;
}

// Gets the color target identified by the given name or adds one according to the given description.
const beGraphics::Any::ColorTextureTarget* Pipe::GetColorTarget(const utf8_ntri &name, const beGraphics::Any::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex, bool *pIsNew)
{
	ColorTarget &target = *GetOrAddTarget<ColorTarget>(m_colorTargets, name);

	uint4 outputMask = 1 << outputIndex;

	if (pIsNew)
		*pIsNew = (target.pTarget == nullptr) || (~target.used & outputMask);

	if (!target.pTarget && (~target.flags & PipeTargetFlags::Immutable | flags & PipeTargetFlags::Flash))
	{
		target.pTarget = NewColorTarget(desc, flags);
		target.flags = flags;
	}
	
	target.used |= outputMask;

	return target.pTarget;
}

// Gets the depth-stencil target identified by the given name or adds one according to the given description.
const beGraphics::Any::DepthStencilTextureTarget* Pipe::GetDepthStencilTarget(const utf8_ntri &name, const beGraphics::Any::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex, bool *pIsNew)
{
	DepthStencilTarget &target = *GetOrAddTarget<DepthStencilTarget>(m_depthStencilTargets, name);

	uint4 outputMask = 1 << outputIndex;

	if (pIsNew)
		*pIsNew = (target.pTarget == nullptr) || (~target.used & outputMask);

	if (!target.pTarget && (~target.flags & PipeTargetFlags::Immutable | flags & PipeTargetFlags::Flash))
	{
		target.pTarget = NewDepthStencilTarget(desc, flags);
		target.flags = flags;
	}
	
	target.used |= outputMask;

	return target.pTarget;
}

// Resets all pipe contents.
void Pipe::Reset(const beGraphics::Any::TextureTargetDesc &desc)
{
	m_bKeepResults = false;

	for (color_target_vector::iterator it = m_colorTargets.begin(); it != m_colorTargets.end(); ++it)
	{
		ColorTarget &target = *it;

		target.pTarget = nullptr;
		target.flags = 0;
		target.used = 0;
	}

	for (depth_stencil_target_vector::iterator it = m_depthStencilTargets.begin(); it != m_depthStencilTargets.end(); ++it)
	{
		DepthStencilTarget &target = *it;

		target.pTarget = nullptr;
		target.flags = 0;
		target.used = 0;
	}

	SetDesc(desc);
}

// Releases all non-permanent pipe contents.
void Pipe::Release()
{
	if (!m_bKeepResults)
	{
		for (color_target_vector::iterator it = m_colorTargets.begin(); it != m_colorTargets.end(); ++it)
		{
			ColorTarget &target = *it;

			if (~target.flags & PipeTargetFlags::Persistent)
			{
				target.pTarget = nullptr;
				target.flags = 0;
				target.used = 0;
			}
		}
		
		for (depth_stencil_target_vector::iterator it = m_depthStencilTargets.begin(); it != m_depthStencilTargets.end(); ++it)
		{
			DepthStencilTarget &target = *it;

			if (~target.flags & PipeTargetFlags::Persistent)
			{
				target.pTarget = nullptr;
				target.flags = 0;
				target.used = 0;
			}
		}
	}
}

// Instructs the pipe to keep its results on release.
void Pipe::KeepResults(bool bKeep)
{
	m_bKeepResults = bKeep;
}

// (Re)sets the final target.
void Pipe::SetFinalTarget(const beGraphics::Any::Texture *pFinalTarget)
{
	SetColorTarget("FinalTarget", nullptr, PipeTargetFlags::Flash, 0);

	if (pFinalTarget)
	{
		const beGraphics::Any::Texture2D &texture = beGraphics::Any::ToTex<beGraphics::TextureType::Texture2D>(*pFinalTarget);

		lean::com_ptr<ID3D11Device> pDevice;
		texture->GetDevice(pDevice.rebind());

		lean::com_ptr<ID3D11ShaderResourceView> pTextureView;
		BE_LOG_DX_ERROR_MSG(
			pDevice->CreateShaderResourceView(texture, nullptr, pTextureView.rebind()),
			"ID3D11Device::CreateShaderResourceView()");

		lean::com_ptr<ID3D11RenderTargetView> pTargetView;
		BE_THROW_DX_ERROR_MSG(
			pDevice->CreateRenderTargetView(texture, nullptr, pTargetView.rebind()),
			"ID3D11Device::CreateRenderTargetView()");

		beGraphics::Any::TextureTargetDesc desc = GetTargetDescTemplate(texture);

		m_pFinalTarget.reset( new beGraphics::Any::ColorTextureTarget(
			desc,
			texture,
			pTextureView,
			pTargetView) );

		SetColorTarget("FinalTarget",
			m_pFinalTarget.get(),
			PipeTargetFlags::Flash | PipeTargetFlags::Immutable | PipeTargetFlags::Persistent, 0);
		SetDesc(desc);
	}
	else
		m_pFinalTarget.reset(nullptr);
}

// Gets the target identified by the given name or nullptr if none available.
const beGraphics::Any::TextureTarget* Pipe::GetFinalTarget() const
{
	return GetAnyTarget("FinalTarget");
}

// (Re)sets the description.
void Pipe::SetDesc(const beGraphics::Any::TextureTargetDesc &desc)
{
	*m_desc = desc;
}

/// Sets a viewport.
void Pipe::SetViewport(const beGraphics::Viewport &viewport)
{
	m_viewport = viewport;
}

} // namespace

} // namespace
