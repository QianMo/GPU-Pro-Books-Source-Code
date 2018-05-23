/*****************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/DX11/beEffectCache.h"
#include "beGraphics/DX11/beEffect.h"
#include "beGraphics/DX11/beTextureCache.h"
#include "beGraphics/DX11/beDevice.h"
#include "beGraphics/DX/beEffect.h"
#include "beGraphics/DX/beIncludeManager.h"

#include <lean/smart/com_ptr.h>
#include <lean/smart/cloneable_obj.h>
#include <unordered_map>
#include <vector>
#include <lean/containers/simple_vector.h>
#include <lean/containers/dynamic_array.h>
#include <lean/smart/scoped_ptr.h>

#include <lean/containers/simple_queue.h>
#include <deque>

#include <beCore/beResourceManagerImpl.hpp>
#include <beCore/beResourceIndex.h>
#include <beCore/beFileWatch.h>

#include <Effects11Lite/D3DEffectsLiteHooks.h>

#include <lean/strings/hashing.h>

#include <lean/io/raw_file.h>
#include <lean/io/mapped_file.h>
#include <lean/io/filesystem.h>

#include <lean/logging/errors.h>
#include <lean/logging/log.h>

extern template beg::DX11::TextureCache::ResourceManagerImpl;
extern template beg::DX11::TextureCache::FiledResourceManagerImpl;

namespace beGraphics
{

namespace DX11
{

/// Effect cache implementation
struct EffectCache::M
{
	lean::cloneable_obj<beCore::PathResolver> resolver;
	lean::cloneable_obj<beCore::ContentProvider> provider;

	EffectCache *cache;
	lean::com_ptr<api::Device> device;
	lean::resource_ptr<TextureCache> pTextureCache;
	utf8_string cacheDir;

	struct Info : public beCore::FileObserver
	{
		lean::resource_ptr<Effect> effect;
		M *m;

		utf8_string resolvedFile;
		utf8_string unresolvedFile;
		typedef lean::dynamic_array<char> macro_backing_store;
		macro_backing_store macroStore;
		typedef lean::dynamic_array<D3D_SHADER_MACRO> macro_vector;
		macro_vector macros;
		typedef lean::dynamic_array<uint4> hook_vector;
		hook_vector hooks;

		/// Constructor.
		Info(Effect *effect, M *m)
			: effect(effect),
			m(m) { }

		/// Method called whenever an observed effect has changed.
		void FileChanged(const lean::utf8_ntri &file, lean::uint8 revision);
	};

	typedef beCore::ResourceIndex<API::Effect, Info> resources_t;
	resources_t resourceIndex;

	typedef lean::simple_vector<lean::scoped_ptr<utf8_t[]>, lean::vector_policies::semipod> hook_vector;
	typedef std::unordered_map< utf8_nt, uint4, lean::hash<utf8_nt> > hook_hash_map;
	hook_vector hooks;
	hook_vector unresolvedHooks;
	hook_hash_map hookHashes;

	beCore::FileWatch fileWatch;
	typedef lean::simple_queue< std::deque< std::pair< lean::resource_ptr<Effect>, lean::resource_ptr<Effect> > > > replace_queue_t;
	replace_queue_t replaceQueue;
	lean::resource_ptr<beCore::ComponentMonitor> pComponentMonitor;

	/// Constructor.
	M(EffectCache *cache, api::Device *device, TextureCache *pTextureCache, const utf8_ntri &cacheDir, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
		: cache(cache),
		pTextureCache(pTextureCache),
		device(device),
		cacheDir( lean::canonical_path<utf8_string>(cacheDir) ),
		resolver(resolver),
		provider(contentProvider)
	{
		LEAN_ASSERT(device != nullptr);
	}
};

namespace
{

/// Appends all files passed to the store vector.
struct VectorIncludeTracker : public DX::IncludeTracker
{
	typedef std::vector<utf8_string> file_vector;
	file_vector *Files;	///< Files.

	/// Constructor.
	VectorIncludeTracker(file_vector *files)
		: Files(files) { }

	/// Called for each actual include file, passing the resolved file & revision.
	void Track(const utf8_ntri &file)
	{
		if (Files)
			Files->push_back(file.to<utf8_string>());
	}
};

/// Gets the name of the corresponding cache file.
utf8_string GetCacheFile(const EffectCache::M &m, const lean::utf8_ntri &file)
{
	return lean::append_path<utf8_string>(
		m.cacheDir,
		lean::get_filename<utf8_string>(file).append(".fxc") );
}

/// Gets the name of the corresponding cache dependency file.
utf8_string GetDependencyFile(const lean::utf8_ntri &cacheFile)
{
	return cacheFile.c_str() + utf8_string(".deps");
}

/// Gets the revision of all dependencies stored in the given dependency file.
uint8 GetDependencyRevision(EffectCache::M &m, const lean::utf8_ntri &dependencyFile, std::vector<utf8_string> *pIncludeFiles)
{
	uint8 dependencyRevision = 0;

	try
	{
		lean::com_ptr<beCore::Content> pDependencyData = m.provider->GetContent(dependencyFile);

		const utf8_t *dependencies = reinterpret_cast<const utf8_t*>( pDependencyData->Bytes() );
		const utf8_t *dependenciesBase = dependencies;
		const utf8_t *dependenciesEnd = dependencies + pDependencyData->Size() / sizeof(utf8_t);

		while (dependencies != dependenciesEnd)
		{
			// Read up to end or next zero delimiter
			while (dependencies != dependenciesEnd && *dependencies)
				++dependencies;

			// Ignore empty strings
			if (dependencies != dependenciesBase)
			{
				// NOTE: might not be null-terminated => construct string
				beCore::Exchange::utf8_string path = m.resolver->Resolve( utf8_string(dependenciesBase, dependencies), false );

				// Check if dependency still existent
				if (!path.empty())
				{
					dependencyRevision = max( dependencyRevision, m.provider->GetRevision(path) );

					if (pIncludeFiles)
						pIncludeFiles->push_back( utf8_string(path.begin(), path.end()) );
				}
			}

			// Move on to next dependency string
			if (dependencies != dependenciesEnd)
				dependenciesBase = ++dependencies;
		}
	}
	catch (...)
	{
		LEAN_LOG_ERROR_CTX("Could not open cached effect dependency file", dependencyFile.c_str());
	}

	return dependencyRevision;
}

struct IncludeManagerEL : D3DEffectsLite::Include
{
	beGraphics::DX::IncludeManager &manager;

	IncludeManagerEL(beGraphics::DX::IncludeManager &manager)
		: manager(manager) { }

	/// Opens the given include file.
	HRESULT D3DEFFECTSLITE_STDCALL Open(D3DEffectsLiteIncludeType type, const char *fileName,
		const void *parent, const void **child, UINT *childSize)
	{
		return manager.Open(D3D_INCLUDE_LOCAL, fileName, parent, child, childSize);
	}
	/// Closes the given include file.
	void D3DEFFECTSLITE_STDCALL Close(const void *child)
	{
		manager.Close(child);
	}
};

// Compiles and caches the given effect.
lean::com_ptr<ID3DBlob, true> CompileAndCacheEffect(EffectCache::M &m, const lean::utf8_ntri &file, const D3D_SHADER_MACRO *pMacros,
		const uint4 *hooks, uint4 hookCount,
		const lean::utf8_ntri &cacheFile, const lean::utf8_ntri &dependencyFile,
		const lean::utf8_ntri &unresolvedFile, std::vector<utf8_string> *pIncludeFiles)
{
	typedef std::vector<utf8_string> file_vector;
	file_vector rawDependencies;

	// Track includes & raw dependencies
	VectorIncludeTracker includeTracker(pIncludeFiles);
	VectorIncludeTracker rawDependencyTracker(&rawDependencies);
	DX::IncludeManager includeManager(*m.resolver, *m.provider, &includeTracker, &rawDependencyTracker);
	
	lean::com_ptr<bec::Content> rawContent = m.provider->GetContent(file);

	// Extract hashed hook files
	lean::dynamic_array<const char*> hookFiles(hookCount);
	for (uint4 i = 0; i < hookCount; ++i)
		hookFiles.push_back(m.unresolvedHooks[i]);

	// Track main file
	includeTracker.Track(file);
	rawDependencyTracker.Track(unresolvedFile);

	// Apply hooks
	IncludeManagerEL includeManagerEL(includeManager);
	lean::com_ptr<D3DEffectsLite::Blob> hookedContent = D3DEffectsLite::HookEffect(rawContent->Data(), (UINT) rawContent->Size(),
		&includeManagerEL, &hookFiles[0], hookCount, file.c_str(),
		"#pragma warning( disable : 3078 )");

	// Compile effect
	lean::com_ptr<ID3DBlob> pData = CompileEffect((const char*) hookedContent->Data(), hookedContent->Size(), file, pMacros, &includeManager);

	try
	{
		{
			// Cache compiled effect
			lean::mapped_file mappedFile(cacheFile, pData->GetBufferSize(), true, lean::file::overwrite, lean::file::sequential);
			memcpy(mappedFile.data(), pData->GetBufferPointer(), pData->GetBufferSize());
		}

		try
		{
			// Dump dependencies
			lean::raw_file rawDependencyFile(dependencyFile, lean::file::write, lean::file::overwrite, lean::file::sequential);

			for (file_vector::const_iterator it = rawDependencies.begin(); it != rawDependencies.end(); ++it)
				// NOTE: Include zero delimiters
				rawDependencyFile.write(it->c_str(), it->size() + 1);
		}
		catch (...)
		{
			LEAN_LOG_ERROR_CTX(
				"Failed to dump dependencies effect to cache",
				dependencyFile.c_str() );
		}
	}
	catch (...)
	{
		LEAN_LOG_ERROR_CTX(
			"Failed to write compiled effect to cache",
			cacheFile.c_str() );
	}

	return pData.transfer();
}

// Re-compiles the given effect.
lean::com_ptr<ID3DX11Effect, true> RecompileEffect(EffectCache::M &m, const lean::utf8_ntri &file, const D3D_SHADER_MACRO *pMacros, const uint4 *hooks, uint4 hookCount,
		const lean::utf8_ntri &mangledFile, const lean::utf8_ntri &unresolvedFile, std::vector<utf8_string> *pIncludeFiles)
{
	utf8_string cacheFile = GetCacheFile(m, mangledFile);
	utf8_string dependencyFile = GetDependencyFile(cacheFile);

	if (pIncludeFiles)
		pIncludeFiles->clear();

	lean::com_ptr<ID3DBlob> pCompiledData = CompileAndCacheEffect(m, file, pMacros, hooks, hookCount,
		cacheFile, dependencyFile, unresolvedFile, pIncludeFiles);
	return CreateEffect(pCompiledData, m.device);
}

// Compiles or loads the given effect.
lean::com_ptr<ID3DX11Effect, true> CompileOrLoadEffect(EffectCache::M &m, const lean::utf8_ntri &file, const D3D_SHADER_MACRO *pMacros, const uint4 *hooks, uint4 hookCount,
		const lean::utf8_ntri &mangledFile, const lean::utf8_ntri &unresolvedFile, std::vector<utf8_string> *pIncludeFiles)
{
	lean::com_ptr<ID3DX11Effect> pEffect;

	utf8_string cacheFile = GetCacheFile(m, mangledFile);
	utf8_string dependencyFile = GetDependencyFile(cacheFile);

	uint8 fileRevision = m.provider->GetRevision(file);
	uint8 cacheRevision = m.provider->GetRevision(cacheFile);

	// Also walk cache dependencies to detect all relevant changes
	if (cacheRevision >= fileRevision)
		fileRevision = max( fileRevision, GetDependencyRevision(m, dependencyFile, pIncludeFiles) );

	// WARNING: Keep alive until effect has been created
	{
		lean::com_ptr<beCore::Content> pCachedData;
		lean::com_ptr<ID3DBlob> pCompiledData;

		const char *pEffectData = nullptr;
		uint4 effectDataSize = 0;

		// Load from cache, if up-to-date, ...
		if (cacheRevision >= fileRevision)
		{
			try
			{
				pCachedData = m.provider->GetContent(cacheFile);
				pEffectData = pCachedData->Bytes();
				effectDataSize = static_cast<uint4>( pCachedData->Size() );
			}
			catch (...)
			{
				LEAN_LOG_ERROR_CTX("Error while trying to load cached effect", cacheFile.c_str());
			}
		}

		// ... recompile otherwise
		if (!pEffectData)
		{
			pCompiledData = CompileAndCacheEffect(m, file, pMacros, hooks, hookCount, cacheFile, dependencyFile, unresolvedFile, pIncludeFiles);
			pEffectData = static_cast<const char*>( pCompiledData->GetBufferPointer() );
			effectDataSize = static_cast<uint4>(pCompiledData->GetBufferSize());
		}

		pEffect = CreateEffect(pEffectData, effectDataSize, m.device);
	}

	return pEffect.transfer();
}

} // namespace

// Constructor.
EffectCache::EffectCache(ID3D11Device *device, TextureCache *pTextureCache, const utf8_ntri &cacheDir, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
	: m( new M(this, device, pTextureCache, cacheDir, resolver, contentProvider) )
{
}

// Destructor.
EffectCache::~EffectCache()
{
}

/// Constructs a new resource info for the given texture.
LEAN_INLINE EffectCache::M::Info MakeResourceInfo(EffectCache::M &m, beg::Effect *effect, EffectCache *cache)
{
	return EffectCache::M::Info(ToImpl(effect), &m);
}

/// Gets the resource from the given resource index iterator.
template <class Iterator>
LEAN_INLINE Effect* GetResource(const EffectCache::M&, Iterator it)
{
	return it->effect;
}

/// Sets the resource for the given resource index iterator.
template <class Iterator>
LEAN_INLINE void SetResource(EffectCache::M&, Iterator it, beg::Effect *resource)
{
	it->effect = ToImpl(resource);
}

/// Gets the resource key for the given resource.
LEAN_INLINE API::Effect* GetResourceKey(const EffectCache::M&, const beg::Effect *pResource)
{
	return (pResource) ? ToImpl(pResource)->Get() : nullptr;
}

namespace
{

/// Gets the mangled file name.
inline utf8_string GetMangledFilename(const lean::utf8_ntri &path, const EffectMacro *pMacros, size_t macroCount, const EffectHook *pHooks, size_t hookCount)
{
	utf8_string mangledFile;
	
	// Decorate path
	if (pMacros && macroCount > 0 || pHooks && hookCount > 0)
	{
		beCore::Exchange::utf8_string mangled = MangleFilename(path, pMacros, macroCount, pHooks, hookCount);
		mangledFile.assign(mangled.begin(), mangled.end());
	}
	else
		mangledFile.assign(path.begin(), path.end());

	return mangledFile;
}

/// Adds the given effect.
inline EffectCache::M::resources_t::file_iterator AddEffect(EffectCache::M &m, const utf8_ntri &unresolvedFile, const utf8_ntri &path, const utf8_string &mangledFile,
	EffectCache::M::Info::macro_vector &macros, EffectCache::M::Info::macro_backing_store &macroStore, 
	EffectCache::M::Info::hook_vector &hooks)
{
	LEAN_FREE_PIMPL(EffectCache);
	M::resources_t::file_iterator itEffect;

	typedef std::vector<utf8_string> file_vector;
	file_vector includeFiles;

	// WARNING: Moved data invalid after transfer
	{
		LEAN_LOG("Attempting to load effect \"" << mangledFile << "\"");

		lean::resource_ptr<Effect> effect = new_resource Effect( 
				CompileOrLoadEffect(m, path, &macros[0], &hooks[0], (uint4) hooks.size(), mangledFile, unresolvedFile, &includeFiles).get(),
				m.pTextureCache
			);

		LEAN_LOG("Effect \"" << unresolvedFile.c_str() << "\" created successfully");

		// Insert effect into cache
		M::resources_t:: iterator rit = m.resourceIndex.Insert(
				*effect,
				m.resourceIndex.GetUniqueName( lean::io::get_stem<utf8_string>(unresolvedFile) ),
				M::Info(effect, &m)
			);
		effect->SetCache(m.cache);
		itEffect = m.resourceIndex.SetFile(rit, mangledFile);
		rit->resolvedFile.assign(path.begin(), path.end());
		rit->unresolvedFile.assign(unresolvedFile.begin(), unresolvedFile.end());
		rit->macroStore = LEAN_MOVE(macroStore);
		rit->macros = LEAN_MOVE(macros);
		rit->hooks = LEAN_MOVE(hooks);
	}

	// Watch entire include graph
	for (file_vector::const_iterator itFile = includeFiles.begin(); itFile != includeFiles.end(); ++itFile)
		m.fileWatch.AddObserver(*itFile, &*itEffect);

	return itEffect;
}

inline lean::scoped_ptr<utf8_t[], lean::critical_ref> AllocateString(utf8_ntri str)
{
	size_t size = str.length() + 1;
	lean::scoped_ptr<utf8_t[]> hookString( new utf8_t[size] );
	memcpy(hookString, str.begin(), sizeof(utf8_t) * size);
	return hookString.transfer();
}

uint4 AddHook(EffectCache::M &m, const EffectHook &hook)
{
	utf8_string unresolvedHook = lean::from_range<utf8_string>(hook.File);
	beCore::Exchange::utf8_string resolvedHook = m.resolver->Resolve(unresolvedHook, true);

	EffectCache::M::hook_hash_map::iterator it = m.hookHashes.find(utf8_nt(resolvedHook));

	if (it == m.hookHashes.end())
	{
		uint4 hash = static_cast<uint4>(m.hooks.size());

		try
		{
			m.unresolvedHooks.push_back().reset( AllocateString(unresolvedHook).detach() );
			m.hooks.push_back().reset( AllocateString(resolvedHook).detach() );
		}
		LEAN_ASSERT_NOEXCEPT

		it = m.hookHashes.insert(std::make_pair(utf8_nt(m.hooks.back().get()), hash)).first;
	}

	return it->second;
}

uint4 GetHook(const EffectCache::M::hook_hash_map &hookHashes, utf8_ntri hook)
{
	EffectCache::M::hook_hash_map::const_iterator it = hookHashes.find(utf8_nt(hook));
	return (it != hookHashes.end()) ? it->second : -1;
}

} // namespace

// Gets the given effect compiled using the given options from file.
Effect* EffectCache::GetByFile(const lean::utf8_ntri &unresolvedFile, const EffectMacro *pMacros, uint4 macroCount, const EffectHook *pHooks, uint4 hookCount)
{
	LEAN_PIMPL();

	// Get absolute path
	beCore::Exchange::utf8_string path = m.resolver->Resolve(unresolvedFile, true);

	// Try to find cached effect
	utf8_string mangledFile = GetMangledFilename(path, pMacros, macroCount, pHooks, hookCount);
	M::resources_t::file_iterator it = m.resourceIndex.FindByFile(mangledFile);

	if (it == m.resourceIndex.EndByFile())
	{
		M::Info::macro_backing_store macroStore;
		M::Info::macro_vector macros = DX::ToAPI( pMacros, (pMacros) ? macroCount : 0, macroStore );
		
		M::Info::hook_vector hooks(hookCount);
		for (size_t i = 0; i < hookCount; ++i)
			hooks.push_back( AddHook(m, pHooks[i]) );

		it = AddEffect(
				m, unresolvedFile, path, mangledFile,
				macros, macroStore, hooks
			);
	}

	return it->effect;
}

// Gets the given effect compiled using the given options from file.
Effect* EffectCache::GetByFile(const lean::utf8_ntri &unresolvedFile, const utf8_ntri &macroString, const utf8_ntri &hookString)
{
	lean::dynamic_array<EffectMacro> macros = DX::ToMacros(macroString);
	lean::dynamic_array<EffectHook> hooks = DX::ToHooks(hookString);

	return GetByFile(unresolvedFile, &macros[0], macros.size(), &hooks[0], hooks.size());
}

/// The file associated with the given resource has changed.
LEAN_INLINE void ResourceFileChanged(EffectCache::M &m, EffectCache::M::resources_t::iterator it, const utf8_ntri &newFile, const utf8_ntri &oldFile)
{
	// Watch texture changes
	if (!newFile.empty())
		m.fileWatch.AddObserver(newFile, &*it);
}

// Gets the given effect compiled using the given options from file, if it has been loaded.
Effect* EffectCache::IdentifyEffect(const lean::utf8_ntri &file, const utf8_ntri &macroString, const utf8_ntri &hookString) const
{
	LEAN_PIMPL_CONST();

	// Get absolute path
	beCore::Exchange::utf8_string path = m.resolver->Resolve(file, true);
	
	lean::dynamic_array<EffectMacro> macros = DX::ToMacros(macroString);
	lean::dynamic_array<EffectHook> hooks = DX::ToHooks(hookString);

	// Try to find cached effect
	utf8_string mangledFile = GetMangledFilename(path, &macros[0], macros.size(), &hooks[0], hooks.size());
	M::resources_t::const_file_iterator it = m.resourceIndex.FindByFile(mangledFile);

	return (it != m.resourceIndex.EndByFile()) ? it->effect : nullptr;
}

// Sets the component monitor.
void EffectCache::SetComponentMonitor(beCore::ComponentMonitor *componentMonitor)
{
	m->pComponentMonitor = componentMonitor;
}

// Gets the component monitor.
beCore::ComponentMonitor* EffectCache::GetComponentMonitor() const
{
	return m->pComponentMonitor;
}

// Gets the file of the given effect.
utf8_ntr EffectCache::GetFile(const beGraphics::Effect *pEffect) const
{
	LEAN_PIMPL_CONST();

	M::resources_t::const_iterator it = m.resourceIndex.Find( GetResourceKey(m, pEffect) );
	return (it != m.resourceIndex.End())
		? utf8_ntr(it->resolvedFile)
		: utf8_ntr("");
}

// Gets the file (or name) of the given effect.
void EffectCache::GetParameters(const beg::Effect *pEffect, beCore::Exchange::utf8_string *pMacros, beCore::Exchange::utf8_string *pHooks) const
{
	LEAN_PIMPL_CONST();

	M::resources_t::const_iterator it = m.resourceIndex.Find( GetResourceKey(m, pEffect) );

	if (it != m.resourceIndex.End())
	{
		if (pMacros)
		{
			pMacros->reserve( it->macroStore.size() );
			DX::ToString(&it->macros.front(), &it->macros.back() + 1, *pMacros);
		}

		if (pHooks)
		{
			bool bFirstHook = true;

			for (M::Info::hook_vector::const_iterator itHook = it->hooks.begin(), itHookEnd = it->hooks.end(); itHook != itHookEnd; ++itHook)
			{
				if (!bFirstHook)
					pHooks->append(1, ';');
				bFirstHook = false;

				pHooks->append(m.unresolvedHooks[*itHook]);
			}
		}
	}
}

// Gets the parameters of the given effect.
void EffectCache::GetParameters(const beGraphics::Effect *effect,
	beCore::Exchange::vector_t<EffectMacro>::t *pMacros, beCore::Exchange::vector_t<EffectHook>::t *pHooks) const
{
	LEAN_PIMPL_CONST();

	M::resources_t::const_iterator it = m.resourceIndex.Find( GetResourceKey(m, effect) );

	if (it != m.resourceIndex.End())
	{
		if (pMacros)
		{
			size_t offset = pMacros->size(), count = it->macroStore.size();
			pMacros->resize(offset + count);

			for (size_t i = 0; i < count; ++i)
			{
				D3D10_SHADER_MACRO const &sm = it->macros[i];
				EffectMacro &dm = (*pMacros)[offset + i];
				dm.Name = utf8_ntr(sm.Name);
				dm.Definition = utf8_ntr(sm.Definition);
			}
		}

		if (pHooks)
		{
			size_t offset = pHooks->size(), count = it->hooks.size();
			pHooks->resize(offset + count);

			for (size_t i = 0; i < count; ++i)
				(*pHooks)[offset + i].File = utf8_ntr(m.unresolvedHooks[it->hooks[i]]);
		}
	}
}

// Checks if the given effects are cache-equivalent.
bool EffectCache::Equivalent(const beGraphics::Effect &left, const beGraphics::Effect &right, bool bIgnoreMacros) const
{
	LEAN_PIMPL_CONST();

	const beGraphics::DX11::Effect &leftImpl = ToImpl(left);
	const beGraphics::DX11::Effect &rightImpl = ToImpl(right);

	if (leftImpl.Get() == rightImpl.Get())
		return true;

	M::resources_t::const_iterator itLeftInfo = m.resourceIndex.Find(leftImpl);
	M::resources_t::const_iterator itRightInfo = m.resourceIndex.Find(rightImpl);

	if (itLeftInfo != m.resourceIndex.End() && itRightInfo != m.resourceIndex.End())
	{
		if (itLeftInfo == itRightInfo)
			return true;
		else if (bIgnoreMacros)
			return (itLeftInfo->resolvedFile == itRightInfo->resolvedFile);
	}

	return false;
}

// Commits changes / reacts to changes.
void EffectCache::Commit()
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
		m.pComponentMonitor->Replacement.AddChanged(Effect::GetComponentType());
}

// Method called whenever an observed effect has changed.
void EffectCache::M::Info::FileChanged(const lean::utf8_ntri &file, lean::uint8 revision)
{
	LEAN_PIMPL();

	M::resources_t::iterator it = m.resourceIndex.Find(this->effect->Get());
	LEAN_ASSERT(it != m.resourceIndex.End());
	M::Info &info = *it;
	LEAN_ASSERT(&info == this);

	typedef std::vector<utf8_string> file_vector;
	file_vector includeFiles;

	lean::resource_ptr<Effect> newEffect = new_resource Effect( 
			RecompileEffect(
				m, info.resolvedFile, &info.macros[0], &info.hooks[0], (uint4) info.hooks.size(),
				m.resourceIndex.GetFile(it), info.unresolvedFile, &includeFiles
			).get(),
			m.pTextureCache
		);

	m.replaceQueue.push_back( std::make_pair(info.effect, newEffect) );

	// Enhance watched include graph
	for (file_vector::const_iterator itFile = includeFiles.begin(); itFile != includeFiles.end(); ++itFile)
		m.fileWatch.AddObserver(*itFile, this);
}

/// Gets the path resolver.
const beCore::PathResolver& EffectCache::GetPathResolver() const
{
	return m->resolver;
}

} // namespace

// Creates a new effect cache.
lean::resource_ptr<EffectCache, true> CreateEffectCache(const Device &device, TextureCache *pTextureCache, const utf8_ntri &cacheDir, 
	const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
{
	return new_resource DX11::EffectCache(
			ToImpl(device),
			ToImpl(pTextureCache),
			cacheDir,
			resolver,
			contentProvider
		);
}

} // namespace