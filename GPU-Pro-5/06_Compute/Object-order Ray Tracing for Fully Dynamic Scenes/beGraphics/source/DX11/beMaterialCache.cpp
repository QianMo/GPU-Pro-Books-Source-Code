/*****************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/DX11/beMaterialCache.h"
#include "beGraphics/DX11/beMaterial.h"
#include "beGraphics/DX11/beMaterialConfig.h"
#include "beGraphics/DX11/beEffectCache.h"
#include "beGraphics/DX11/beMaterialConfigCache.h"

#include <lean/smart/cloneable_obj.h>
#include <lean/smart/com_ptr.h>

#include <beCore/beResourceManagerImpl.hpp>
#include <beCore/beResourceIndex.h>
#include <beCore/beFileWatch.h>

#include <lean/io/filesystem.h>

#include <lean/logging/log.h>

extern template beg::DX11::EffectCache::ResourceManagerImpl;
extern template beg::DX11::EffectCache::FiledResourceManagerImpl;

extern template beg::DX11::MaterialConfigCache::ResourceManagerImpl;
// extern template beg::DX11::MaterialConfigCache::FiledResourceManagerImpl;

namespace beGraphics
{

namespace DX11
{

/// Texture cache implementation
struct MaterialCache::M // : public beCore::FileObserver
{
	lean::cloneable_obj<beCore::PathResolver> resolver;
	lean::cloneable_obj<beCore::ContentProvider> provider;

	MaterialCache *cache;
	lean::resource_ptr<EffectCache> effectCache;
	lean::resource_ptr<MaterialConfigCache> configCache;

	struct Info
	{
		lean::resource_ptr<Material> resource;

		/// Constructor.
		Info(Material *resource)
			: resource(resource) { }
	};

	typedef beCore::ResourceIndex<beg::Material, Info> resources_t;
	resources_t resourceIndex;

	beCore::FileWatch fileWatch;

	lean::resource_ptr<beCore::ComponentMonitor> pComponentMonitor;

	/// Constructor.
	M(MaterialCache *cache, EffectCache *effectCache, MaterialConfigCache *configCache, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
		: cache(cache),
		configCache(configCache),
		effectCache(effectCache),
		resolver(resolver),
		provider(contentProvider)
	{
		LEAN_ASSERT(configCache != nullptr);
	}

	/// Method called whenever an observed texture has changed.
//	void FileChanged(const lean::utf8_ntri &file, lean::uint8 revision) LEAN_OVERRIDE;
};

// Constructor.
MaterialCache::MaterialCache(EffectCache *effectCache, MaterialConfigCache *configCache,
							 const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
	: m( new M(this, effectCache, configCache, resolver, contentProvider) )
{
}

// Destructor.
MaterialCache::~MaterialCache()
{
}

/// Constructs a new resource info for the given texture.
LEAN_INLINE MaterialCache::M::Info MakeResourceInfo(MaterialCache::M &m, beg::Material *material, MaterialCache *)
{
	return MaterialCache::M::Info(ToImpl(material));
}

/// Sets the resource for the given resource index iterator.
template <class Iterator>
LEAN_INLINE void SetResource(MaterialCache::M&, Iterator it, beg::Material *resource)
{
	it->resource = ToImpl(resource);
}

// Gets a texture from the given file.
beGraphics::Material* MaterialCache::NewByFile(const lean::utf8_ntri &unresolvedFile, const lean::utf8_ntri &name)
{
	LEAN_PIMPL();

	beg::Effect *mainEffect = m.effectCache->GetByFile(unresolvedFile);
	lean::resource_ptr<Material> material = new_resource Material(&mainEffect, 1, *m.effectCache);

	utf8_string uniqueName = m.resourceIndex.GetUniqueName( name.empty() ? lean::get_stem<utf8_string>(unresolvedFile) : name );
	SetName(material, uniqueName);

	return material;
}

// Sets the component monitor.
void MaterialCache::SetComponentMonitor(beCore::ComponentMonitor *componentMonitor)
{
	m->pComponentMonitor = componentMonitor;
}

// Gets the component monitor.
beCore::ComponentMonitor* MaterialCache::GetComponentMonitor() const
{
	return m->pComponentMonitor;
}

/// Gets the path resolver.
const beCore::PathResolver& MaterialCache::GetPathResolver() const
{
	return m->resolver;
}

// Commits changes / reacts to changes.
void MaterialCache::Commit()
{
	LEAN_PIMPL();

	if (!m.pComponentMonitor ||
		!m.pComponentMonitor->Replacement.HasChanged(Effect::GetComponentType()) &&
		!m.pComponentMonitor->Replacement.HasChanged(MaterialConfig::GetComponentType()))
		return;

	bool bHasChanges = false;

	std::vector<const beg::Effect*> newEffects;
	std::vector<beg::MaterialConfig*> newConfigs;

	for (M::resources_t::iterator it = m.resourceIndex.Begin(), itEnd = m.resourceIndex.End(); it != itEnd; ++it)
	{
		Material *material = it->resource;
		lean::resource_ptr<Material> newMaterial = material;

		Material::Effects effects = material->GetEffects();
		newEffects.assign(effects.begin(), effects.end());
		bool bEffectsChanged = false;

		for (size_t i = 0, count = newEffects.size(); i < count; ++i)
			while (beg::Effect *successor = newEffects[i]->GetSuccessor())
			{
				newEffects[i] = successor;
				bEffectsChanged = true;
			}

		// IMPORTANT: Also monitor linked effects
		for (Material::Effects linkedEffects = material->GetLinkedEffects(); linkedEffects; ++linkedEffects)
			bEffectsChanged |= (linkedEffects[0]->GetSuccessor() != nullptr);

		Material::Configurations configs = material->GetConfigurations();
		newConfigs.assign(configs.begin(), configs.end());
		bool bConfigsChanged = false;

		for (size_t i = 0, count = newConfigs.size(); i < count; ++i)
			while (beg::MaterialConfig *successor = newConfigs[i]->GetSuccessor())
			{
				newConfigs[i] = successor;
				bConfigsChanged = true;
			}

		if (bEffectsChanged)
			newMaterial = new_resource Material(&newEffects[0], (uint4) newEffects.size(), *m.effectCache);
		if (bEffectsChanged || bConfigsChanged)
			newMaterial->SetConfigurations(&newConfigs[0], (uint4) newConfigs.size());

		if (newMaterial != material)
			Replace(material, newMaterial);
		bHasChanges |= bEffectsChanged | bConfigsChanged;
	}

	// Notify dependents
	if (bHasChanges && m.pComponentMonitor)
		m.pComponentMonitor->Replacement.AddChanged(Material::GetComponentType());
}

} // namespace

// Creates a new texture cache.
lean::resource_ptr<MaterialCache, lean::critical_ref> CreateMaterialCache(EffectCache *effectCache, MaterialConfigCache *configCache,
																		  const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
{
	return new_resource DX11::MaterialCache(ToImpl(effectCache), ToImpl(configCache), resolver, contentProvider);
}

} // namespace