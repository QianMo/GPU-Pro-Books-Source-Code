/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beMeshCompound.h"
#include "beScene/beMesh.h"

#include <beCore/beReflectionProperties.h>

#include <lean/io/numeric.h>

namespace beScene
{

template <class Material>
struct MeshProperties
{
	// MONITOR: REDUNDANT: Must specify size here
	static const beCore::ReflectionProperty value[1];
};

template <class Material>
const beCore::ReflectionProperty MeshProperties<Material>::value[] =
{
	beCore::MakeReflectionProperty<float>("lod dist", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&MeshCompound<Material>::SetLODDistance) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&MeshCompound<Material>::GetLODDistance) )
};

// Constructor,
template <class Material>
MeshCompound<Material>::MeshCompound()
{
}

// Constructor,
template <class Material>
MeshCompound<Material>::MeshCompound(Mesh *const *meshBegin, Mesh *const *meshEnd,
		Material *const *materialBegin, Material *const *materialEnd,
		const LOD *lodBegin, const LOD *lodEnd)
	: m_meshes(meshBegin, meshEnd),
	m_materials(materialBegin, materialEnd),
	m_lods(lodBegin, lodEnd)
{
	// Fill incomplete subsets with null materials
	m_materials.resize(m_meshes.size());

	// By default move all meshes into one level of detail
	if (m_lods.empty())
	{
		m_lods.resize(1);
		m_lods[0].Subsets.End = static_cast<uint4>(m_meshes.size());
	}

	uint4 nextLodSubset = 0;

	// Verify LOD integrity
	for (lod_vector::iterator lod = m_lods.begin(); lod != m_lods.end(); ++lod)
	{
		// Consecutive ranges of subsets required
		LEAN_ASSERT(lod->Subsets.Begin == nextLodSubset);
		LEAN_ASSERT(lod->Subsets.Begin <= lod->Subsets.End);
		// MONITOR: Throw instead?!

		nextLodSubset = lod->Subsets.End;
	}

	// Make sure all meshes are contained
	LEAN_ASSERT(nextLodSubset == m_meshes.size());
}

// Destructor.
template <class Material>
MeshCompound<Material>::~MeshCompound()
{
}

namespace
{

template <class LOD, class Mesh>
LOD& GetLOD(std::vector<LOD> &lods, uint4 &lodIdx, const std::vector<Mesh> &meshes)
{
	// Accept -1 for last level of detail
	if (lodIdx == -1)
		lodIdx = static_cast<uint4>(lods.size() - 1);

	// Append sufficient number of levels
	if (lodIdx >= lods.size())
	{
		lods.resize(
			lodIdx + 1,
			LOD(
				bec::MakeRangeN((uint4) meshes.size(), 0),
				(!lods.empty()) ? lods.back().Distance : 0.0f
			) );
	}

	return lods[lodIdx];
}

} // namespace

// Adds the given mesh setting the given material.
template <class Material>
uint4 MeshCompound<Material>::AddMeshWithMaterial(Mesh *mesh, Material *pMaterial, uint4 lodIdx)
{
	LEAN_ASSERT(mesh);

	LOD &lod = GetLOD(m_lods, lodIdx, m_meshes);

	// Append the subset to the given level of detail
	uint4 subsetIdx = lod.Subsets.End;
	m_meshes.insert(m_meshes.begin() + subsetIdx, mesh);
	try { m_materials.insert(m_materials.begin() + subsetIdx, pMaterial); }
	catch (...) { LEAN_ASSERT_UNREACHABLE(); }
	++lod.Subsets.End;

	// Update subsequent levels of detail
	for (uint4 i = lodIdx + 1; i < m_lods.size(); ++i)
	{
		++m_lods[i].Subsets.Begin;
		++m_lods[i].Subsets.End;
	}

	// TODO: Emit changed?!

	return subsetIdx;
}

// Sets the n-th mesh.
template <class Material>
void MeshCompound<Material>::SetMeshWithMaterial(uint4 subsetIdx, Mesh *pMesh, Material *pMaterial)
{
	if (subsetIdx < m_meshes.size())
	{
		// Update what has been given
		if (pMesh)
			 m_meshes[subsetIdx] = pMesh;
		if (pMaterial)
			m_materials[subsetIdx] = pMaterial;

		// TODO: Emit changed?!
	}
	else
		// Append to last level of detail
		AddMeshWithMaterial(pMesh, pMaterial, -1);
}

// Removes the given mesh with the given material.
template <class Material>
void MeshCompound<Material>::RemoveMeshWithMaterial(Mesh *pMesh, Material *pMaterial)
{
	uint4 removedCount = 0;

	for (lod_vector::iterator lod = m_lods.begin(); lod != m_lods.end(); ++lod)
	{
		// Update level of detail with all subsets removed in previous levels
		lod->Subsets.Begin -= removedCount;
		lod->Subsets.End -= removedCount;

		for (uint4 i = lod->Subsets.End; i-- > lod->Subsets.Begin; )
			// Erase matching pairs of mesh and material
			if (m_meshes[i] == pMesh && (!pMaterial || m_materials[i] == pMaterial))
			{
				// Erase subset
				m_meshes.erase(m_meshes.begin() + i);
				m_materials.erase(m_materials.begin() + i);

				// Update current AND subsequent levels
				--lod->Subsets.End;
				++removedCount;
			}
	}

	// TODO: Emit changed?!
}

// Removes the n-th subset.
template <class Material>
void MeshCompound<Material>::RemoveSubset(uint4 subsetIdx)
{
	if (subsetIdx < m_meshes.size())
	{
		// Erase subset
		m_meshes.erase(m_meshes.begin() + subsetIdx);
		m_materials.erase(m_materials.begin() + subsetIdx);

		// Update current and subsequent levels of detail
		for (lod_vector::iterator lod = m_lods.begin(); lod != m_lods.end(); ++lod)
		{
			if  (lod->Subsets.Begin > subsetIdx)
				--lod->Subsets.Begin;
			if  (lod->Subsets.End > subsetIdx)
				--lod->Subsets.End;
		}

		// TODO: Emit changed?!
	}
}

// Sets the distance for the given level of detail.
template <class Material>
void MeshCompound<Material>::SetLODDistance(uint4 lodIdx, float distance)
{
	GetLOD(m_lods, lodIdx, m_meshes).Distance = distance;
}

// Sets the given distances for all levels of detail.
template <class Material>
uint4 MeshCompound<Material>::SetLODDistance(const float *distances, uint4 count)
{
	count = min(count, (uint4) m_lods.size());
	for (uint4 i = 0; i < count; ++i)
		m_lods[i].Distance = distances[i];
	return count;
}

// Gets the distances for all levels of detail.
template <class Material>
uint4 MeshCompound<Material>::GetLODDistance(float *distances, uint4 count) const
{
	count = min(count, (uint4) m_lods.size());
	for (uint4 i = 0; i < count; ++i)
		distances[i] = m_lods[i].Distance;
	return count;
}

/// Gets the type of the given property.
template <class Material>
PropertyDesc MeshCompound<Material>::GetPropertyDesc(uint4 id) const
{
	// Widen all properties to LOD count
	PropertyDesc desc = bec::ReflectionPropertyProvider::GetPropertyDesc(id);
	desc.Count = (uint4) m_lods.size();
	return desc;
}

// Gets the reflection properties.
template <class Material>
typename MeshCompound<Material>::Properties MeshCompound<Material>::GetReflectionProperties() const
{
	return ToPropertyRange(MeshProperties<Material>::value);
}

// Gets the reflection properties.
template <class Material>
typename MeshCompound<Material>::Properties MeshCompound<Material>::GetOwnProperties()
{
	return ToPropertyRange(MeshProperties<Material>::value);
}

// Gets the number of child components.
template <class Material>
uint4 MeshCompound<Material>::GetComponentCount() const
{
	return static_cast<uint4>( m_meshes.size() );
}

// Gets the name of the n-th child component.
template <class Material>
beCore::Exchange::utf8_string MeshCompound<Material>::GetComponentName(uint4 idx) const
{
	beCore::Exchange::utf8_string name;

	LEAN_ASSERT(idx < m_meshes.size());
	const utf8_string &meshName = m_meshes[idx]->GetName();
	utf8_string subsetNum = lean::int_to_string(idx);
	
	bool bNamed = !meshName.empty();
	name.reserve(
			lean::ntarraylen("Subset ") + subsetNum.size()
			 + meshName.size() + lean::ntarraylen(" ()") * bNamed
		);

	name.append(meshName.begin(), meshName.end());
	if (bNamed)
		name.append(" (");
	name.append("Subset ");
	name.append(subsetNum.begin(), subsetNum.end());
	if (bNamed)
		name.append(")");

	return name;
}

// Gets the n-th reflected child component, nullptr if not reflected.
template <class Material>
lean::com_ptr<const beCore::ReflectedComponent, lean::critical_ref> MeshCompound<Material>::GetReflectedComponent(uint4 idx) const
{
	LEAN_ASSERT(idx < m_meshes.size());
	return Reflect(m_materials[idx]);
}

// Gets the type of the n-th child component.
template <class Material>
const beCore::ComponentType* MeshCompound<Material>::GetComponentType(uint4 idx) const
{
	return Material::GetComponentType();
}

// Gets the n-th component.
template <class Material>
lean::cloneable_obj<lean::any, true> MeshCompound<Material>::GetComponent(uint4 idx) const
{
	return bec::any_resource_t<Material>::t( const_cast<Material*>( m_materials[idx].get() ) );
}

// Returns true, if the n-th component can be replaced.
template <class Material>
bool MeshCompound<Material>::IsComponentReplaceable(uint4 idx) const
{
	return true;
}

// Sets the n-th component.
template <class Material>
void MeshCompound<Material>::SetComponent(uint4 idx, const lean::any &pComponent)
{
	SetMeshWithMaterial( idx, nullptr, any_cast<Material*>(pComponent) );
}

} // namespace