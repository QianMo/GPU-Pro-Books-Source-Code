/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beBoundMaterialCache.h"
#include <beCore/beComponentMonitor.h>

#include <lean/smart/resource_ptr.h>
#include <unordered_map>

#include <lean/logging/errors.h>
#include <lean/logging/log.h>

namespace beScene
{

/// Renderable material cache implementation
struct GenericBoundMaterialCache::M
{
	typedef std::unordered_map<beg::Material*, lean::resource_ptr<GenericBoundMaterial>> materials_t;
	materials_t materials;

	lean::resource_ptr<beCore::ComponentMonitor> pComponentMonitor;
};

// Constructor.
GenericBoundMaterialCache::GenericBoundMaterialCache()
	: m( new M() )
{
}

// Destructor.
GenericBoundMaterialCache::~GenericBoundMaterialCache()
{
}

// Gets a material from the given effect & name.
GenericBoundMaterial* GenericBoundMaterialCache::GetMaterial(beg::Material *pMaterial)
{
	LEAN_PIMPL();

	if (!pMaterial)
		return nullptr;

	// Get cached material
	M::materials_t::iterator itMaterial = m.materials.find(pMaterial);
	
	if (itMaterial == m.materials.end())
		itMaterial = m.materials.insert(std::make_pair( pMaterial, CreateBoundMaterial(pMaterial) )).first;

	return itMaterial->second;
}

// Sets the component monitor.
void GenericBoundMaterialCache::SetComponentMonitor(beCore::ComponentMonitor *componentMonitor)
{
	m->pComponentMonitor = componentMonitor;
}

// Gets the component monitor.
beCore::ComponentMonitor* GenericBoundMaterialCache::GetComponentMonitor() const
{
	return m->pComponentMonitor;
}

// Commits / reacts to changes.
void GenericBoundMaterialCache::Commit()
{
	LEAN_PIMPL();

	if (!m.pComponentMonitor || !m.pComponentMonitor->Replacement.HasChanged(beg::Material::GetComponentType()))
		return;

	bool bHasChanges = false;

	for (M::materials_t::iterator it = m.materials.end(); it-- != m.materials.begin(); )
	{
		beg::Material *newMaterial = it->first;

		// Check for new materials
		while (beg::Material *successor = newMaterial->GetSuccessor())
			newMaterial = successor;

		if (newMaterial != it->first)
		{
			// Replace old bound material by new one
			GenericBoundMaterial* newBoundMaterial = GetMaterial(newMaterial);
			it->second->SetSuccessor(newBoundMaterial);
			bHasChanges = true;

			// Release old binding, unlikely to be needed again
			it = m.materials.erase(it);
		}
	}

	// Notify dependents
	if (bHasChanges && m.pComponentMonitor)
		m.pComponentMonitor->Replacement.AddChanged(this->GetComponentType());
}

} // namespace
