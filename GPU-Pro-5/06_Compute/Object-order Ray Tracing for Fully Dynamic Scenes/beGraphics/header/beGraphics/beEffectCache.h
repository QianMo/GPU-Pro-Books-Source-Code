/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_EFFECT_CACHE
#define BE_GRAPHICS_EFFECT_CACHE

#include "beGraphics.h"
#include <beCore/beShared.h>
#include <beCore/beResourceManager.h>
#include <lean/tags/noncopyable.h>
#include "beEffect.h"
#include <beCore/bePathResolver.h>
#include <beCore/beContentProvider.h>
#include <beCore/beComponentMonitor.h>
#include <lean/smart/resource_ptr.h>

namespace beGraphics
{

namespace Exchange = beCore::Exchange;

/// Effect macro.
struct EffectMacro
{
	lean::range<const utf8_t*> Name;		///< Macro name.
	lean::range<const utf8_t*> Definition;	///< Macro definition.

	EffectMacro() { }
	/// Constructor.
	EffectMacro(const lean::utf8_ntr &name, const lean::utf8_ntr &definition)
		: Name(name),
		Definition(definition) { }
	/// Constructor.
	EffectMacro(const lean::range<const utf8_t*> &name, const lean::range<const utf8_t*> &definition)
		: Name(name),
		Definition(definition) { }
};

/// Effect hook.
struct EffectHook
{
	lean::range<const utf8_t*> File;		///< Hook file name.

	EffectHook() { }
	/// Constructor.
	explicit EffectHook(const lean::utf8_ntr &file)
		: File(file) { }
	/// Constructor.
	explicit EffectHook(const lean::range<const utf8_t*> &file)
		: File(file) { }
};

/// Effect cache.
class LEAN_INTERFACE EffectCache : public lean::noncopyable, public beCore::FiledResourceManager<Effect>, public Implementation
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(EffectCache)

public:
	/// Gets the given effect compiled using the given options from file.
	virtual Effect* GetByFile(const lean::utf8_ntri &file,
		const EffectMacro *pMacros = nullptr, uint4 macroCount = 0,
		const EffectHook *pHooks = nullptr, uint4 hookCount = 0) = 0;
	/// Gets the given effect compiled using the given options from file.
	virtual Effect* GetByFile(const lean::utf8_ntri &file, const utf8_ntri &macros, const utf8_ntri &hooks) = 0;

	/// Gets the parameters of the given effect.
	virtual void GetParameters(const Effect *effect,
		beCore::Exchange::utf8_string *pMacros = nullptr, beCore::Exchange::utf8_string *pHooks = nullptr) const = 0;
	/// Gets the parameters of the given effect.
	virtual void GetParameters(const Effect *effect,
		beCore::Exchange::vector_t<EffectMacro>::t *pMacros = nullptr, beCore::Exchange::vector_t<EffectHook>::t *pHooks = nullptr) const = 0;

	/// Gets the given effect compiled using the given options from file, if it has been loaded.
	virtual Effect* IdentifyEffect(const lean::utf8_ntri &file, const utf8_ntri &macros, const utf8_ntri &hooks) const = 0;
	/// Checks if the given effects are cache-equivalent.
	virtual bool Equivalent(const Effect &left, const Effect &right, bool bIgnoreMacros = false) const = 0;

	/// Sets the component monitor.
	virtual void SetComponentMonitor(beCore::ComponentMonitor *componentMonitor) = 0;
	/// Gets the component monitor.
	virtual beCore::ComponentMonitor* GetComponentMonitor() const = 0;

	/// Gets the path resolver.
	virtual const beCore::PathResolver& GetPathResolver() const = 0;
};

/// Mangles the given file name & macros.
BE_GRAPHICS_API Exchange::utf8_string MangleFilename(const lean::utf8_ntri &file,
													 const EffectMacro *pMacros, size_t macroCount,
													 const EffectHook *pHooks, size_t hookCount);

// Prototypes
class Device;
class TextureCache;

/// Creates a new effect cache.
BE_GRAPHICS_API lean::resource_ptr<EffectCache, true> CreateEffectCache(const Device &device, TextureCache *pTextureCache, const utf8_ntri &cacheDir,
	const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider);

}

#endif