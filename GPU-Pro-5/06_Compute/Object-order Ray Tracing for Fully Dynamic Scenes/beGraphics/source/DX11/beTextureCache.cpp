/*****************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/DX11/beTextureCache.h"
#include "beGraphics/DX11/beTexture.h"
#include "beGraphics/DX11/beD3D11.h"
#include "beGraphics/DX11/beDevice.h"

#include <lean/smart/cloneable_obj.h>
#include <lean/smart/com_ptr.h>
#include <lean/containers/simple_queue.h>
#include <deque>

#include <beCore/beResourceManagerImpl.hpp>
#include <beCore/beResourceIndex.h>
#include <beCore/beFileWatch.h>

#include <lean/io/filesystem.h>

#include <lean/logging/log.h>

namespace beGraphics
{

namespace DX11
{

/// Texture cache implementation
struct TextureCache::M : public beCore::FileObserver
{
	lean::cloneable_obj<beCore::PathResolver> resolver;
	lean::cloneable_obj<beCore::ContentProvider> provider;

	TextureCache *cache;
	lean::com_ptr<api::Device> device;

	struct Info
	{
		lean::resource_ptr<Texture> texture;
		lean::resource_ptr<TextureView> pTextureView;

		bool bSRGB;

		/// Constructor.
		Info(Texture *texture, bool bSRGB)
			: texture(texture),
			bSRGB(bSRGB) { }
	};

	typedef beCore::ResourceIndex<API::Resource, Info> resources_t;
	resources_t resourceIndex;

	beCore::FileWatch fileWatch;
	typedef lean::simple_queue< std::deque< std::pair< lean::resource_ptr<Texture>, lean::resource_ptr<Texture> > > > replace_queue_t;
	replace_queue_t replaceQueue;
	lean::resource_ptr<beCore::ComponentMonitor> pComponentMonitor;

	/// Constructor.
	M(TextureCache *cache, api::Device *device, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
		: cache(cache),
		device(device),
		resolver(resolver),
		provider(contentProvider)
	{
		LEAN_ASSERT(device != nullptr);
	}

	/// Method called whenever an observed texture has changed.
	void FileChanged(const lean::utf8_ntri &file, lean::uint8 revision) LEAN_OVERRIDE;
};

// Constructor.
TextureCache::TextureCache(api::Device *device, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
	: m( new M(this, device, resolver, contentProvider) )
{
}

// Destructor.
TextureCache::~TextureCache()
{
}

/// Constructs a new resource info for the given texture.
LEAN_INLINE TextureCache::M::Info MakeResourceInfo(TextureCache::M &m, beg::Texture *texture, TextureCache*)
{
	return TextureCache::M::Info(ToImpl(texture), false);
}

/// Gets the resource from the given resource index iterator.
template <class Iterator>
LEAN_INLINE Texture* GetResource(const TextureCache::M&, Iterator it)
{
	return it->texture;
}

/// Sets the resource for the given resource index iterator.
template <class Iterator>
LEAN_INLINE void SetResource(TextureCache::M &m, Iterator it, beg::Texture *resource)
{
	LEAN_FREE_PIMPL(TextureCache);

	M::Info &info = *it;
	info.texture = ToImpl(resource);

	// IMPORTANT: Keep texture view in sync
	if (info.pTextureView)
	{
		if (info.pTextureView->ref_count() > 1)
		{
			lean::resource_ptr<TextureView> newView = new_resource TextureView(info.texture->GetResource(), nullptr, m.device);
			newView->SetCache(m.cache);
			info.pTextureView->SetSuccessor(newView);
			info.pTextureView = newView;
		}
		else
			info.pTextureView = nullptr;
	}
}

/// Gets the resource key for the given resource.
LEAN_INLINE API::Resource* GetResourceKey(const TextureCache::M&, const beg::Texture *pResource)
{
	return (pResource) ? ToImpl(pResource)->GetResource() : nullptr;
}

/// Default resource change monitoring implementation. Replace using ADL.
template <class Iterator>
LEAN_INLINE void ResourceChanged(TextureCache::M &m, Iterator it)
{
	if (m.pComponentMonitor)
		m.pComponentMonitor->Replacement.AddChanged(beg::TextureView::GetComponentType());
}

/// Default resource management change monitoring implementation. Replace using ADL.
template <class Iterator>
LEAN_INLINE void ResourceManagementChanged(TextureCache::M &m, Iterator it)
{
	if (m.pComponentMonitor)
		m.pComponentMonitor->Management.AddChanged(beg::TextureView::GetComponentType());
}

// Loads a texture from the given file.
lean::com_ptr<ID3D11Resource, true> LoadTexture(TextureCache::M &m, const lean::utf8_ntri &file, bool bSRGB)
{
	lean::com_ptr<beCore::Content> content = m.provider->GetContent(file);
	return DX11::LoadTexture(m.device, content->Bytes(), static_cast<uint4>(content->Size()), nullptr, bSRGB);
}

// Gets a texture from the given file.
beGraphics::Texture* TextureCache::GetByFile(const lean::utf8_ntri &unresolvedFile, bool bSRGB)
{
	LEAN_PIMPL();

	// Get absolute path
	beCore::Exchange::utf8_string excPath = m.resolver->Resolve(unresolvedFile, true);
	utf8_string path(excPath.begin(), excPath.end());

	// Try to find cached resource
	M::resources_t::file_iterator it = m.resourceIndex.FindByFile(path);

	if (it == m.resourceIndex.EndByFile())
	{
		LEAN_LOG("Attempting to load texture \"" << path << "\"");
		lean::resource_ptr<Texture> pTexture = CreateTexture( LoadTexture(m, path, bSRGB).get() );
		LEAN_LOG("Texture \"" << unresolvedFile.c_str() << "\" created successfully");

		// Insert texture into cache
		M::resources_t::iterator rit = m.resourceIndex.Insert(
				pTexture->GetResource(),
				m.resourceIndex.GetUniqueName(lean::get_stem<utf8_string>(unresolvedFile)),
				M::Info(pTexture, bSRGB)
			);
		pTexture->SetCache(this);
		it = m.resourceIndex.SetFile(rit, path);
		
		// Watch texture changes
		m.fileWatch.AddObserver(path, &m);
	}

	return it->texture;
}

/// The file associated with the given resource has changed.
LEAN_INLINE void ResourceFileChanged(TextureCache::M &m, TextureCache::M::resources_t::iterator it, const utf8_ntri &newFile, const utf8_ntri &oldFile)
{
	// Watch texture changes
	if (!oldFile.empty())
		m.fileWatch.RemoveObserver(oldFile, &m);
	if (!newFile.empty())
		m.fileWatch.AddObserver(newFile, &m);
}

// Gets a texture for the given texture view.
beGraphics::Texture* TextureCache::GetTexture(const beGraphics::TextureView *pView) const
{
	LEAN_PIMPL_CONST();

	API::Resource *pTextureDX = (pView) ? ToImpl(pView)->GetResource() : nullptr;
	M::resources_t::const_iterator it = m.resourceIndex.Find(pTextureDX);

	return (it != m.resourceIndex.End())
		? it->texture
		: nullptr;
}

// Gets a texture view for the given texture.
beGraphics::TextureView* TextureCache::GetView(const beGraphics::Texture *pTexture)
{
	LEAN_PIMPL();
	TextureView *pView = nullptr;

	API::Resource *pTextureDX = (pTexture) ? ToImpl(pTexture)->GetResource() : nullptr;
	M::resources_t::iterator it = m.resourceIndex.Find(pTextureDX);

	if (it != m.resourceIndex.End())
	{
		if (!it->pTextureView)
		{
			it->pTextureView = new_resource TextureView(pTextureDX, nullptr, m.device);
			it->pTextureView->SetCache(this);
		}

		pView = it->pTextureView;
	}

	return pView;
}

// Gets whether the given texture is an srgb texture.
bool TextureCache::IsSRGB(const beGraphics::Texture *pTexture) const
{
	LEAN_PIMPL_CONST();
	M::resources_t::const_iterator it = m.resourceIndex.Find((pTexture) ? ToImpl(pTexture)->GetResource() : nullptr);

	return (it != m.resourceIndex.End())
		? it->bSRGB
		: false;
}

// Sets the component monitor.
void TextureCache::SetComponentMonitor(beCore::ComponentMonitor *componentMonitor)
{
	m->pComponentMonitor = componentMonitor;
}

// Gets the component monitor.
beCore::ComponentMonitor* TextureCache::GetComponentMonitor() const
{
	return m->pComponentMonitor;
}

// Commits changes / reacts to changes.
void TextureCache::Commit()
{
	LEAN_PIMPL();

	bool bHasChanges = !m.replaceQueue.empty();

	while (!m.replaceQueue.empty())
	{
		M::replace_queue_t::value_type replacePair = m.replaceQueue.pop_front();
		Replace(replacePair.first, replacePair.second);
	}

	// Notify dependents
	if (bHasChanges && m.pComponentMonitor)
		m.pComponentMonitor->Replacement.AddChanged(TextureView::GetComponentType());
}

// Method called whenever an observed texture has changed.
void TextureCache::M::FileChanged(const lean::utf8_ntri &file, lean::uint8 revision)
{
	LEAN_STATIC_PIMPL();

	M::resources_t::file_iterator it = m.resourceIndex.FindByFile(file.to<utf8_string>());

	if (it != m.resourceIndex.EndByFile())
	{
		M::Info &info = *it;

		lean::resource_ptr<Texture> newTexture = CreateTexture( LoadTexture(m, file, info.bSRGB).get() );
		
		m.replaceQueue.push_back( std::make_pair(info.texture, newTexture) );
	}
}

/// Gets the path resolver.
const beCore::PathResolver& TextureCache::GetPathResolver() const
{
	return m->resolver;
}

} // namespace

// Creates a new texture cache.
lean::resource_ptr<TextureCache, true> CreateTextureCache(const Device &device, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
{
	return new_resource DX11::TextureCache(ToImpl(device), resolver, contentProvider);
}

} // namespace