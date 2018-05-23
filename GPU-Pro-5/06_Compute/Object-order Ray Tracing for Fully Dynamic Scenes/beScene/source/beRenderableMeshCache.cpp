/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beRenderableMeshCache.h"
#include "beScene/beRenderableMaterial.h"

#include "beScene/beRenderableMesh.h"
#include "beScene/beAssembledMesh.h"

#include <beGraphics/DX11/beDevice.h>

#include <beCore/beResourceIndex.h>
#include <beCore/beResourceManagerImpl.hpp>

#include <lean/logging/errors.h>
#include <lean/logging/log.h>

namespace beScene
{

/// Mesh cache implementation
struct RenderableMeshCache::M
{
	RenderableMeshCache *cache;
	lean::resource_ptr<beg::Device> device;

	struct Info
	{
		lean::resource_ptr<RenderableMesh> resource;

		Info(RenderableMesh *resource)
			: resource(resource) { }
	};

	typedef bec::ResourceIndex<besc::RenderableMesh, Info> resources_t;
	resources_t resourceIndex;

	lean::resource_ptr<beCore::ComponentMonitor> pComponentMonitor;

	/// Constructor.
	M(RenderableMeshCache *cache, beGraphics::Device *device)
		: cache(cache),
		device(device)
	{
		LEAN_ASSERT(device != nullptr);
	}
};

// Constructor.
RenderableMeshCache::RenderableMeshCache(beGraphics::Device *device)
	: m( new M(this, device) )
{
}

// Destructor.
RenderableMeshCache::~RenderableMeshCache()
{
}

// Commits / reacts to changes.
void RenderableMeshCache::Commit()
{
	LEAN_PIMPL();

	if (!m.pComponentMonitor ||
		!m.pComponentMonitor->Replacement.HasChanged(RenderableMaterial::GetComponentType()) &&
		!m.pComponentMonitor->Replacement.HasChanged(AssembledMesh::GetComponentType()))
		return;

	bool bMeshHasChanges = false;
	bool bDataHasChanges = false;

	for (M::resources_t::iterator it = m.resourceIndex.Begin(), itEnd = m.resourceIndex.End(); it != itEnd; ++it)
	{
		RenderableMesh *mesh = it->resource;

		if (const AssembledMesh *oldSource = mesh->GetSource())
		{
			const AssembledMesh *newSource = bec::GetSuccessor(oldSource);

			if (newSource != oldSource)
			{
				lean::resource_ptr<RenderableMesh> newMesh = ToRenderableMesh(*newSource, nullptr, true);
				// TODO: Replace assembled mesh material configs
				TransferMaterials(*mesh, *newMesh);
				Replace(mesh, newMesh);
				mesh = newMesh;
				bMeshHasChanges = true;
			}

		}

		RenderableMesh::MaterialRange materials = mesh->GetMaterials();
		for (uint4 i = 0, count = Size4(materials); i < count; ++i)
		{
			RenderableMaterial *material = bec::GetSuccessor(materials[i]);
			if (material != materials[i])
			{
				mesh->SetMeshWithMaterial(i, nullptr, material);
				bDataHasChanges = true;
			}
		}
	}

	// Notify dependents
	if (bMeshHasChanges)
		m.pComponentMonitor->Replacement.AddChanged(RenderableMesh::GetComponentType());
	if (bDataHasChanges)
		m.pComponentMonitor->Data.AddChanged(RenderableMesh::GetComponentType());
}

// Sets the component monitor.
void RenderableMeshCache::SetComponentMonitor(beCore::ComponentMonitor *componentMonitor)
{
	m->pComponentMonitor = componentMonitor;
}

// Gets the component monitor.
beCore::ComponentMonitor* RenderableMeshCache::GetComponentMonitor() const
{
	return m->pComponentMonitor;
}

// Creates a new mesh cache.
lean::resource_ptr<RenderableMeshCache, true> CreateRenderableMeshCache(beGraphics::Device *device)
{
	return new_resource RenderableMeshCache(device);
}

} // namespace