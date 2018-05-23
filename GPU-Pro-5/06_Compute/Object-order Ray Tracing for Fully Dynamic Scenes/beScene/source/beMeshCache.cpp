/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beMeshCache.h"
#include "beScene/DX11/beMesh.h"
#include "beScene/beMeshSerialization.h"

#include <beGraphics/beDevice.h>

#include <beCore/beResourceIndex.h>
#include <beCore/beResourceManagerImpl.hpp>
#include <beCore/beFileWatch.h>

#include <lean/smart/cloneable_obj.h>
#include <lean/smart/com_ptr.h>
#include <lean/containers/simple_queue.h>
#include <deque>

#include <lean/io/filesystem.h>

#include <lean/logging/errors.h>
#include <lean/logging/log.h>

namespace beScene
{

/// Mesh cache implementation
struct MeshCache::M : public beCore::FileObserver
{
	lean::cloneable_obj<beCore::PathResolver> resolver;
	lean::cloneable_obj<beCore::ContentProvider> provider;

	MeshCache *cache;
	lean::resource_ptr<beg::Device> device;

	struct Info
	{
		lean::resource_ptr<AssembledMesh> resource;

		Info(AssembledMesh *resource)
			: resource(resource) { }
	};

	typedef bec::ResourceIndex<besc::AssembledMesh, Info> resources_t;
	resources_t resourceIndex;

	beCore::FileWatch fileWatch;
	typedef lean::simple_queue< std::deque< std::pair< lean::resource_ptr<AssembledMesh>, lean::resource_ptr<AssembledMesh> > > > replace_queue_t;
	replace_queue_t replaceQueue;
	lean::resource_ptr<beCore::ComponentMonitor> pComponentMonitor;

	/// Constructor.
	M(MeshCache *cache, beGraphics::Device *device, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
		: resolver(resolver),
		provider(contentProvider),
		cache(cache),
		device(device)
	{
		LEAN_ASSERT(device != nullptr);
	}

	/// Method called whenever an observed mesh has changed.
	void FileChanged(const lean::utf8_ntri &file, lean::uint8 revision) LEAN_OVERRIDE;
};

// Loads a mesh from the given file.
lean::resource_ptr<AssembledMesh, true> LoadMesh(MeshCache::M &m, const lean::utf8_ntri &file)
{
	lean::com_ptr<beCore::Content> content = m.provider->GetContent(file);
	return LoadMeshes( content->Bytes(), content->Size(), *m.device );
}

// Constructor.
MeshCache::MeshCache(beGraphics::Device *device, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
	: m( new M(this, device, resolver, contentProvider) )
{
}

// Destructor.
MeshCache::~MeshCache()
{
}

// Adds a named mesh.
besc::AssembledMesh* MeshCache::SetName(beScene::Mesh *mesh, const lean::utf8_ntri &name)
{
	LEAN_THROW_NULL(mesh);
	lean::resource_ptr<besc::AssembledMesh> compound = new_resource beScene::AssembledMesh(
		&mesh, &mesh + 1, nullptr, nullptr, nullptr, nullptr);
	SetName(compound, name);
	return compound;
}

// Gets a mesh from the given file.
beScene::AssembledMesh* MeshCache::GetByFile(const lean::utf8_ntri &unresolvedFile)
{
	LEAN_PIMPL();

	// Get absolute path
	beCore::Exchange::utf8_string excPath = m.resolver->Resolve(unresolvedFile, true);
	utf8_string path(excPath.begin(), excPath.end());

	// Try to find cached mesh
	M::resources_t::file_iterator it = m.resourceIndex.FindByFile(path);

	if (it == m.resourceIndex.EndByFile())
	{
		LEAN_LOG("Attempting to load mesh \"" << path << "\"");
		lean::resource_ptr<AssembledMesh> mesh = LoadMesh(m, path);
		LEAN_LOG("Mesh \"" << unresolvedFile.c_str() << "\" created successfully");

		// Insert mesh into cache
		M::resources_t::iterator rit = m.resourceIndex.Insert(
				mesh,
				m.resourceIndex.GetUniqueName( lean::get_stem<utf8_string>(unresolvedFile) ),
				M::Info(mesh)
			);
		mesh->SetCache(this);
		it = m.resourceIndex.SetFile(rit, path);

		// Watch mesh changes
		m.fileWatch.AddObserver(path, &m);
	}

	return it->resource;
}

/// The file associated with the given resource has changed.
LEAN_INLINE void ResourceFileChanged(MeshCache::M &m, MeshCache::M::resources_t::iterator it, const utf8_ntri &newFile, const utf8_ntri &oldFile)
{
	// Watch texture changes
	if (!oldFile.empty())
		m.fileWatch.RemoveObserver(oldFile, &m);
	if (!newFile.empty())
		m.fileWatch.AddObserver(newFile, &m);
}

// Sets the component monitor.
void MeshCache::SetComponentMonitor(beCore::ComponentMonitor *componentMonitor)
{
	m->pComponentMonitor = componentMonitor;
}

// Gets the component monitor.
beCore::ComponentMonitor* MeshCache::GetComponentMonitor() const
{
	return m->pComponentMonitor;
}

// Commits / reacts to changes.
void MeshCache::Commit()
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
		m.pComponentMonitor->Replacement.AddChanged(AssembledMesh::GetComponentType());
}

// Method called whenever an observed mesh has changed.
void MeshCache::M::FileChanged(const lean::utf8_ntri &file, lean::uint8 revision)
{
	LEAN_STATIC_PIMPL();

	M::resources_t::file_iterator it = m.resourceIndex.FindByFile(file.to<utf8_string>());

	if (it != m.resourceIndex.EndByFile())
	{
		M::Info &info = *it;

		LEAN_LOG("Attempting to reload mesh \"" << file.c_str() << "\"");
		lean::resource_ptr<AssembledMesh> newResource = LoadMesh(m, file);
		LEAN_LOG("Mesh \"" << file.c_str() << "\" recreated successfully");

		m.replaceQueue.push_back( std::make_pair(info.resource, newResource) );
	}
}

/// Gets the path resolver.
const beCore::PathResolver& MeshCache::GetPathResolver() const
{
	return m->resolver;
}

// Creates a new mesh cache.
lean::resource_ptr<MeshCache, true> CreateMeshCache(beGraphics::Device *device, const beCore::PathResolver &resolver, const beCore::ContentProvider &contentProvider)
{
	return new_resource MeshCache(device, resolver, contentProvider);
}

} // namespace