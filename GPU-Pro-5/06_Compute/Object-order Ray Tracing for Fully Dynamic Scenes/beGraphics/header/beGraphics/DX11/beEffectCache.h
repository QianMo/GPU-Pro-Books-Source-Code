/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_EFFECT_CACHE_DX11
#define BE_GRAPHICS_EFFECT_CACHE_DX11

#include "beGraphics.h"
#include "../beEffectCache.h"
#include <beCore/beResourceManagerImpl.h>
#include "beEffect.h"
#include "beD3DXEffects11.h"
#include <lean/pimpl/pimpl_ptr.h>
#include <lean/smart/resource_ptr.h>

namespace beGraphics
{

namespace DX11
{

class TextureCache;

/// Effect cache implementation.
class EffectCache : public beCore::FiledResourceManagerImpl<beGraphics::Effect, EffectCache, beGraphics::EffectCache>
{
	friend ResourceManagerImpl;
	friend FiledResourceManagerImpl;

public:
	struct M;

private:
	lean::pimpl_ptr<M> m;

public:
	/// Constructor.
	BE_GRAPHICS_DX11_API EffectCache(api::Device *pDevice, TextureCache *pTextureCache, const utf8_ntri &cacheDir,
		const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~EffectCache();

	/// Gets the given effect compiled using the given options from file.
	BE_GRAPHICS_DX11_API Effect* GetByFile(const lean::utf8_ntri &file,
		const EffectMacro *pMacros = nullptr, uint4 macroCount = 0,
		const EffectHook *pHooks = nullptr, uint4 hookCount = 0) LEAN_OVERRIDE;
	/// Gets the given effect compiled using the given options from file.
	BE_GRAPHICS_DX11_API Effect* GetByFile(const lean::utf8_ntri &file, const utf8_ntri &macros, const utf8_ntri &hooks) LEAN_OVERRIDE;

	/// Gets the file of the given effect.
	BE_GRAPHICS_DX11_API utf8_ntr GetFile(const beGraphics::Effect *effect) const LEAN_OVERRIDE;
	/// Gets the parameters of the given effect.
	BE_GRAPHICS_DX11_API void GetParameters(const beGraphics::Effect *effect,
		beCore::Exchange::utf8_string *pMacros = nullptr, beCore::Exchange::utf8_string *pHooks = nullptr) const LEAN_OVERRIDE;
	/// Gets the parameters of the given effect.
	BE_GRAPHICS_DX11_API void GetParameters(const beGraphics::Effect *effect,
		beCore::Exchange::vector_t<EffectMacro>::t *pMacros = nullptr, beCore::Exchange::vector_t<EffectHook>::t *pHooks = nullptr) const LEAN_OVERRIDE;

	/// Gets the given effect compiled using the given options from file, if it has been loaded.
	BE_GRAPHICS_DX11_API Effect* IdentifyEffect(const lean::utf8_ntri &file, const utf8_ntri &macros, const utf8_ntri &hooks) const LEAN_OVERRIDE;
	/// Checks if the given effects are cache-equivalent.
	BE_GRAPHICS_DX11_API bool Equivalent(const beGraphics::Effect &left, const beGraphics::Effect &right, bool bIgnoreMacros = false) const LEAN_OVERRIDE;
	
	/// Commits changes / reacts to changes.
	BE_GRAPHICS_DX11_API void Commit();

	/// Sets the component monitor.
	BE_GRAPHICS_DX11_API void SetComponentMonitor(beCore::ComponentMonitor *componentMonitor) LEAN_OVERRIDE;
	/// Gets the component monitor.
	BE_GRAPHICS_DX11_API beCore::ComponentMonitor* GetComponentMonitor() const LEAN_OVERRIDE;

	/// Gets the path resolver.
	BE_GRAPHICS_DX11_API const beCore::PathResolver& GetPathResolver() const LEAN_OVERRIDE;

	/// Gets the implementation identifier.
	ImplementationID GetImplementationID() const LEAN_OVERRIDE { return DX11Implementation; }
};

template <> struct ToImplementationDX11<beGraphics::EffectCache> { typedef EffectCache Type; };

} // namespace

} // namespace

#endif