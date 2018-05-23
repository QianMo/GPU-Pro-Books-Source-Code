/*****************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/DX11/beMaterialConfigCache.h"
#include "beGraphics/DX11/beMaterialConfig.h"
#include "beGraphics/DX11/beTextureCache.h"

#include <lean/smart/cloneable_obj.h>
#include <lean/smart/com_ptr.h>

#include <beCore/beResourceManagerImpl.hpp>
#include <beCore/beResourceIndex.h>
#include <beCore/beFileWatch.h>

#include <lean/io/filesystem.h>

#include <lean/logging/log.h>

extern template beg::DX11::TextureCache::ResourceManagerImpl;
extern template beg::DX11::TextureCache::FiledResourceManagerImpl;

namespace beGraphics
{

namespace DX11
{

/// Texture cache implementation
struct MaterialConfigCache::M // : public beCore::FileObserver
{
	lean::cloneable_obj<beCore::PathResolver> resolver;
	lean::cloneable_obj<beCore::ContentProvider> provider;

	MaterialConfigCache *cache;
	lean::resource_ptr<TextureCache> textureCache;

	struct Info
	{
		lean::resource_ptr<MaterialConfig> resource;

		/// Constructor.
		Info(MaterialConfig *resource)
			: resource(resource) { }
	};

	typedef beCore::ResourceIndex<beg::MaterialConfig, Info> resources_t;
	resources_t resourceIndex;

	beCore::FileWatch fileWatch;

	lean::resource_ptr<beCore::ComponentMonitor> pComponentMonitor;

	/// Constructor.
	M(MaterialConfigCache *cache, TextureCache *textureCache, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
		: cache(cache),
		textureCache(textureCache),
		resolver(resolver),
		provider(contentProvider)
	{
		LEAN_ASSERT(textureCache != nullptr);
	}

	/// Method called whenever an observed texture has changed.
//	void FileChanged(const lean::utf8_ntri &file, lean::uint8 revision) LEAN_OVERRIDE;
};

// Constructor.
MaterialConfigCache::MaterialConfigCache(TextureCache *textureCache, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
	: m( new M(this, textureCache, resolver, contentProvider) )
{
}

// Destructor.
MaterialConfigCache::~MaterialConfigCache()
{
}

/// Constructs a new resource info for the given texture.
LEAN_INLINE MaterialConfigCache::M::Info MakeResourceInfo(MaterialConfigCache::M &m, beg::MaterialConfig *config, MaterialConfigCache *)
{
	return MaterialConfigCache::M::Info(ToImpl(config));
}

/// Sets the resource for the given resource index iterator.
template <class Iterator>
LEAN_INLINE void SetResource(MaterialConfigCache::M&, Iterator it, beg::MaterialConfig *resource)
{
	it->resource = ToImpl(resource);
}

// Sets the component monitor.
void MaterialConfigCache::SetComponentMonitor(beCore::ComponentMonitor *componentMonitor)
{
	m->pComponentMonitor = componentMonitor;
}

// Gets the component monitor.
beCore::ComponentMonitor* MaterialConfigCache::GetComponentMonitor() const
{
	return m->pComponentMonitor;
}

// Gets the texture cache.
beg::TextureCache* MaterialConfigCache::GetTextureCache() const
{
	return m->textureCache;
}

// Gets the path resolver.
const beCore::PathResolver& MaterialConfigCache::GetPathResolver() const
{
	return m->resolver;
}

// Commits changes / reacts to changes.
void MaterialConfigCache::Commit()
{
	LEAN_PIMPL();

	if (!m.pComponentMonitor ||
		!m.pComponentMonitor->Replacement.HasChanged(TextureView::GetComponentType()))
		return;

	bool bHasChanges = false;

	std::vector<const beg::Effect*> newEffects;
	std::vector<beg::MaterialConfig*> newConfigs;

	for (M::resources_t::iterator it = m.resourceIndex.Begin(), itEnd = m.resourceIndex.End(); it != itEnd; ++it)
	{
		MaterialConfig *material = it->resource;

		for (uint4 i = 0, count = material->GetTextureCount(); i < count; ++i)
		{
			const beg::TextureView *texture = material->GetTexture(i);
			bool bTextureChanged = false;

			while (const beg::TextureView *successor = texture->GetSuccessor())
			{
				texture = successor;
				bTextureChanged = true;
			}

			if (bTextureChanged)
			{
				material->SetTexture(i, texture);
				bHasChanges = true;
			}
		}
	}

	// Notify dependents
	if (bHasChanges && m.pComponentMonitor)
		m.pComponentMonitor->Data.AddChanged(MaterialConfig::GetComponentType());
}

} // namespace

// Creates a new texture cache.
lean::resource_ptr<MaterialConfigCache, lean::critical_ref> CreateMaterialConfigCache(TextureCache *textureCache, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
{
	return new_resource DX11::MaterialConfigCache(ToImpl(textureCache), resolver, contentProvider);
}

} // namespace