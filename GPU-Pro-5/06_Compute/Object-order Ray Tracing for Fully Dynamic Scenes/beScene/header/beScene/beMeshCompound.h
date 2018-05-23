/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_MESH_COMPOUND
#define BE_SCENE_MESH_COMPOUND

#include "beScene.h"
#include <beCore/beShared.h>
#include <beCore/beReflectionPropertyProvider.h>
#include <beCore/beMany.h>
#include <vector>
#include <lean/smart/resource_ptr.h>
#include <lean/tags/noncopyable.h>

namespace beScene
{

class Mesh;

using beCore::PropertyDesc;

struct MeshCompoundBase : public lean::nonassignable
{
	/// Level of detail.
	struct LOD
	{
		beCore::Range<uint4> Subsets;	///< Range of mesh subsets.
		float Distance;					///< Minimum detail distance.

		/// Default constructor.
		LEAN_INLINE LOD()
			: Distance() { }
		/// Initializes this level of detail.
		LEAN_INLINE LOD(beCore::Range<uint4> subsets, float distance)
			: Subsets(subsets),
			Distance(distance) { }
	};
};

/// Mesh compound.
template <class Material>
class MeshCompound : public MeshCompoundBase, 
	public beCore::ResourceAsRefCounted< beCore::PropertyFeedbackProvider<beCore::ReflectionPropertyProvider> >
{
public:
	/// Subset mesh vector.
	typedef std::vector< lean::resource_ptr<Mesh> > mesh_vector;
	/// Subset material vector.
	typedef std::vector< lean::resource_ptr<Material> > material_vector;
	/// LOD vector.
	typedef std::vector< LOD > lod_vector;

private:
	mesh_vector m_meshes;
	material_vector m_materials;
	lod_vector m_lods;

public:
	/// Empty compound constructor.
	BE_SCENE_API MeshCompound();
	/// Full compound constructor. Consecutive levels of detail have to reference consecutive subset ranges.
	BE_SCENE_API MeshCompound(Mesh *const *meshBegin, Mesh *const *meshEnd,
		Material *const *materialBegin, Material *const *materialEnd,
		const LOD *lodBegin, const LOD *lodEnd);
	/// Destructor.
	BE_SCENE_API ~MeshCompound();

	typedef beCore::Range<Mesh* const*> MeshRange;
	typedef beCore::Range<Material* const*> MaterialRange;
	typedef beCore::Range<const LOD*> LODRange;

	/// Gets all meshes.
	LEAN_INLINE MeshRange GetMeshes() const { return beCore::MakeRangeN(&m_meshes[0].get(), m_meshes.size()); }
	/// Gets all materials.
	LEAN_INLINE MaterialRange GetMaterials() const { return beCore::MakeRangeN(&m_materials[0].get(), m_materials.size()); }
	/// Gets all levels of detail.
	LEAN_INLINE LODRange GetLODs() const { return beCore::MakeRangeN(m_lods.data(), m_lods.size()); }

	/// Adds the given mesh setting the given material.
	BE_SCENE_API uint4 AddMeshWithMaterial(Mesh *mesh, Material *pMaterial, uint4 lod = 0);
	/// Sets the n-th mesh.
	BE_SCENE_API void SetMeshWithMaterial(uint4 subsetIdx, Mesh *pMesh, Material *pMaterial);
	/// Removes the given mesh with the given material.
	BE_SCENE_API void RemoveMeshWithMaterial(Mesh *pMesh, Material *pMaterial = nullptr);
	/// Removes the n-th subset.
	BE_SCENE_API void RemoveSubset(uint4 subsetIdx);

	/// Sets the distance for the given level of detail.
	BE_SCENE_API void SetLODDistance(uint4 lod, float distance);

	/// Sets the given distances for all levels of detail.
	BE_SCENE_API uint4 SetLODDistance(const float *distances, uint4 count);
	/// Gets the distances for all levels of detail.
	BE_SCENE_API uint4 GetLODDistance(float *distances, uint4 count) const;

	/// Gets the type of the given property.
	BE_SCENE_API PropertyDesc GetPropertyDesc(uint4 id) const;
	/// Gets the reflection properties.
	BE_SCENE_API Properties GetReflectionProperties() const;
	/// Gets the reflection properties.
	BE_SCENE_API static Properties GetOwnProperties();

	/// Gets the number of child components.
	BE_SCENE_API uint4 GetComponentCount() const;
	/// Gets the name of the n-th child component.
	BE_SCENE_API beCore::Exchange::utf8_string GetComponentName(uint4 idx) const;
	/// Gets the n-th reflected child component, nullptr if not reflected.
	BE_SCENE_API lean::com_ptr<const ReflectedComponent, lean::critical_ref> GetReflectedComponent(uint4 idx) const;

	/// Gets the type of the n-th child component.
	BE_SCENE_API const beCore::ComponentType* GetComponentType(uint4 idx) const;
	/// Gets the n-th component.
	BE_SCENE_API lean::cloneable_obj<lean::any, true> GetComponent(uint4 idx) const;
	/// Returns true, if the n-th component can be replaced.
	BE_SCENE_API bool IsComponentReplaceable(uint4 idx) const;
	/// Sets the n-th component.
	BE_SCENE_API void SetComponent(uint4 idx, const lean::any &pComponent);
};

} // namespace

#endif