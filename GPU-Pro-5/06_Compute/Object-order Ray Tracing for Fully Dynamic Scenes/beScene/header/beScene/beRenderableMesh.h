/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_RENDERABLE_MESH
#define BE_SCENE_RENDERABLE_MESH

#include "beScene.h"
#include "beMeshCompound.h"
#include "beAssembledMesh.h"
#include <beCore/beManagedResource.h>
#include <beGraphics/beMaterialCache.h>

namespace beScene
{

class RenderableMeshCache;
class RenderableMaterial;
class RenderableMaterialCache;

/// Renderable mesh.
class RenderableMesh : public MeshCompound<RenderableMaterial>,
	public beCore::ManagedResource<RenderableMeshCache>, public beCore::HotResource<RenderableMesh>
{
	lean::resource_ptr<const AssembledMesh> m_pSource;

public:
	/// Empty compound constructor.
	LEAN_INLINE RenderableMesh(AssembledMesh *pSource = nullptr)
		: m_pSource(pSource) { }
	/// Full compound constructor. Consecutive levels of detail have to reference consecutive subset ranges.
	LEAN_INLINE RenderableMesh(Mesh *const *meshBegin, Mesh *const *meshEnd,
			RenderableMaterial *const *materialBegin, RenderableMaterial *const *materialEnd,
			const LOD *lodBegin, const LOD *lodEnd,
			const AssembledMesh *pSource = nullptr)
		: MeshCompound<RenderableMaterial>(meshBegin, meshEnd, materialBegin, materialEnd, lodBegin, lodEnd),
		m_pSource(pSource) { }
	/// Copies the given mesh compound.
	LEAN_INLINE RenderableMesh(const RenderableMesh &right)
		: MeshCompound<RenderableMaterial>(right.GetMeshes().Begin, right.GetMeshes().End,
			right.GetMaterials().Begin, right.GetMaterials().End,
			right.GetLODs().Begin, right.GetLODs().End),
		m_pSource(right.m_pSource) { }

	/// Sets the source mesh.
	LEAN_INLINE void SetSource(const AssembledMesh *pSource) { m_pSource = pSource; }
	/// Gets the source mesh.
	LEAN_INLINE const AssembledMesh* GetSource() const { return m_pSource; }

	/// Gets the type.
	BE_SCENE_API static const beCore::ComponentType* GetComponentType();
	/// Gets the type.
	BE_SCENE_API const beCore::ComponentType* GetType() const;
};

/// Adds all meshes in the given assembled mesh to the given renderable mesh using the given material.
BE_SCENE_API lean::resource_ptr<RenderableMesh, lean::critical_ref> ToRenderableMesh(
	const AssembledMesh &src, RenderableMaterial *pMaterial, bool bStoreSource);
/// Fills in missing materials using the given material.
BE_SCENE_API void FillRenderableMesh(RenderableMesh &mesh, RenderableMaterial *material, RenderableMaterialCache &materialCache);
/// Adds all meshes in the given assembled mesh to the given renderable mesh using (and adapting) the given material.
BE_SCENE_API lean::resource_ptr<RenderableMesh, lean::critical_ref> ToRenderableMesh(
	const AssembledMesh &src, RenderableMaterial *material, RenderableMaterialCache &materialCache, bool bStoreSource);
/// Fills in missing materials using the given material.
BE_SCENE_API void CacheMaterials(const RenderableMesh &mesh, beGraphics::MaterialCache &materialCache);
/// Transfers materials from one renderable mesh to another, matching subset mesh names.
BE_SCENE_API void TransferMaterials(const RenderableMesh &source, RenderableMesh &dest);

} // namespace

#endif