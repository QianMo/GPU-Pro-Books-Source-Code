/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/DX11/bePipe.h"

namespace beScene
{

// Gets a new color target matching the given description.
lean::com_ptr<const beGraphics::ColorTextureTarget, true> Pipe::NewColorTarget(const beGraphics::TextureTargetDesc &desc, uint4 flags) const
{
	return ToImpl(this)->NewColorTarget( ToAPI(desc), flags );
}

// Gets a new depth-stencil target matching the given description.
lean::com_ptr<const beGraphics::DepthStencilTextureTarget, true> Pipe::NewDepthStencilTarget(const beGraphics::TextureTargetDesc &desc, uint4 flags) const
{
	return ToImpl(this)->NewDepthStencilTarget( ToAPI(desc), flags );
}

// Gets the target identified by the given name or nullptr if none available.
const beGraphics::TextureTarget* Pipe::GetAnyTarget(const utf8_ntri &name) const
{
	return ToImpl(this)->GetAnyTarget( name );
}

// Gets the color target identified by the given name or nullptr if none available.
const beGraphics::ColorTextureTarget* Pipe::GetColorTarget(const utf8_ntri &name) const
{
	return ToImpl(this)->GetColorTarget( name );
}

// Gets the depth-stencil target identified by the given name nullptr if none available.
const beGraphics::DepthStencilTextureTarget* Pipe::GetDepthStencilTarget(const utf8_ntri &name) const
{
	return ToImpl(this)->GetDepthStencilTarget( name );
}

// Updates the color target identified by the given name.
void Pipe::SetColorTarget(const utf8_ntri &name, const beGraphics::ColorTextureTarget *pTarget, uint4 flags, uint4 outputIndex, 
	lean::com_ptr<const beGraphics::ColorTextureTarget> *pOldTarget)
{
	return ToImpl(this)->SetColorTarget( name, pTarget, flags, outputIndex, pOldTarget );
}

// Updates the color target identified by the given name.
void Pipe::SetDepthStencilTarget(const utf8_ntri &name, const beGraphics::DepthStencilTextureTarget *pTarget, uint4 flags, uint4 outputIndex, 
	lean::com_ptr<const beGraphics::DepthStencilTextureTarget> *pOldTarget)
{
	return ToImpl(this)->SetDepthStencilTarget( name, pTarget, flags, outputIndex, pOldTarget );
}

// Gets a new color target matching the given description and stores it under the given name.
const beGraphics::ColorTextureTarget* Pipe::GetNewColorTarget(const utf8_ntri &name, const beGraphics::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex, 
	lean::com_ptr<const beGraphics::ColorTextureTarget> *pOldTarget)
{
	return ToImpl(this)->GetNewColorTarget( name, ToAPI(desc), flags, outputIndex, pOldTarget );
}

// Gets a new depth-stencil target matching the given description and stores it under the given name.
const beGraphics::DepthStencilTextureTarget* Pipe::GetNewDepthStencilTarget(const utf8_ntri &name, const beGraphics::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex, 
	lean::com_ptr<const beGraphics::DepthStencilTextureTarget> *pOldTarget)
{
	return ToImpl(this)->GetNewDepthStencilTarget( name, ToAPI(desc), flags, outputIndex, pOldTarget );
}

// Gets the color target identified by the given name or adds one according to the given description.
const beGraphics::ColorTextureTarget* Pipe::GetColorTarget(const utf8_ntri &name, const beGraphics::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex, bool *pIsNew)
{
	return ToImpl(this)->GetColorTarget( name, ToAPI(desc), flags, outputIndex, pIsNew );
}

// Gets the depth-stencil target identified by the given name or adds one according to the given description.
const beGraphics::DepthStencilTextureTarget* Pipe::GetDepthStencilTarget(const utf8_ntri &name, const beGraphics::TextureTargetDesc &desc, uint4 flags, uint4 outputIndex, bool *pIsNew)
{
	return ToImpl(this)->GetDepthStencilTarget( name, ToAPI(desc), flags, outputIndex, pIsNew );
}

// Resets all pipe contents.
void Pipe::Reset(const beGraphics::TextureTargetDesc &desc)
{
	ToImpl(this)->Reset( ToAPI(desc) );
}

// Releases all non-permanent pipe contents.
void Pipe::Release()
{
	ToImpl(this)->Release();
}

// Instructs the pipe to keep its results on release.
void Pipe::KeepResults(bool bKeep)
{
	ToImpl(this)->KeepResults( bKeep );
}

// (Re)sets the final target.
void Pipe::SetFinalTarget(const beGraphics::Texture *pFinalTarget)
{
	ToImpl(this)->SetFinalTarget( ToImpl(pFinalTarget) );
}

// Gets the target identified by the given name or nullptr if none available.
const beGraphics::TextureTarget* Pipe::GetFinalTarget() const
{
	return ToImpl(this)->GetFinalTarget();
}

// (Re)sets the description.
void Pipe::SetDesc(const beGraphics::TextureTargetDesc &desc)
{
	ToImpl(this)->SetDesc( ToAPI(desc) );
}

// Gets the description.
beGraphics::TextureTargetDesc Pipe::GetDesc() const
{
	return FromAPI( ToImpl(this)->GetDesc() );
}

/// Sets a viewport.
void Pipe::SetViewport(const beGraphics::Viewport &viewport)
{
	ToImpl(this)->SetViewport( viewport );
}

// Gets the current viewport.
beGraphics::Viewport Pipe::GetViewport() const
{
	return ToImpl(this)->GetViewport();
}

/// Constructor.
lean::resource_ptr<Pipe, true> CreatePipe(const beGraphics::Texture &finalTarget, beGraphics::TextureTargetPool *pTargetPool)
{
	return lean::bind_resource<Pipe>( new DX11::Pipe( ToImpl(finalTarget), ToImpl(pTargetPool) ) );
}

/// Constructor.
lean::resource_ptr<Pipe, true> CreatePipe(const beGraphics::TextureTargetDesc &desc, beGraphics::TextureTargetPool *pTargetPool)
{
	return lean::bind_resource<Pipe>( new DX11::Pipe( ToAPI(desc), ToImpl(pTargetPool) ) );
}

} // namespace
