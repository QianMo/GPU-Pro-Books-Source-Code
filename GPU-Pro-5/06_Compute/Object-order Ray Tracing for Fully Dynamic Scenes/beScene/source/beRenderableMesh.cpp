/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beRenderableMesh.h"

#include "beScene/beMesh.h"

#include "beScene/beRenderableMaterial.h"
#include <beGraphics/beMaterial.h>
#include "beScene/beRenderableMaterialCache.h"

#include <beCore/beComponentTypes.h>

#include <lean/functional/algorithm.h>

namespace beScene
{
	extern const beCore::ComponentType RenderableMeshType = { "RenderableMesh" };

	template MeshCompound<RenderableMaterial>;

	/// Gets the type.
	const beCore::ComponentType* RenderableMesh::GetComponentType()
	{
		return &RenderableMeshType;
	}
	// Gets the type.
	const beCore::ComponentType* RenderableMesh::GetType() const
	{
		return &RenderableMeshType;
	}

} // namespace

#include "beMeshCompound.cpp"

#include "beScene/beAssembledMesh.h"
extern template beScene::MeshCompound<beGraphics::MaterialConfig>;

namespace beScene
{

// Adds all meshes in the given assembled mesh to the given renderable mesh using the given material, optionally cloning & adapting the material.
lean::resource_ptr<RenderableMesh, lean::critical_ref> ToRenderableMesh(const AssembledMesh &src, RenderableMaterial *pMaterial,
																		bool bStoreSource)
{
	lean::resource_ptr<RenderableMesh> mesh = new_resource RenderableMesh(
		src.GetMeshes().Begin, src.GetMeshes().End,
		nullptr, nullptr,
		src.GetLODs().Begin, src.GetLODs().End,
		(bStoreSource) ? &src : nullptr );

	for (uint4 i = 0, count = Size4(mesh->GetMeshes()); i < count; ++i)
		mesh->SetMeshWithMaterial(i, nullptr, pMaterial);

	return mesh.transfer();
}

// Fills in missing materials using the given material.
void FillRenderableMesh(RenderableMesh &mesh, RenderableMaterial *material, RenderableMaterialCache &materialCache)
{
	for (uint4 subsetIdx = 0, count = Size4(mesh.GetMeshes()); subsetIdx < count; ++subsetIdx)
		if (!mesh.GetMaterials()[subsetIdx])
		{
			const Mesh *subsetMesh = mesh.GetMeshes()[subsetIdx];
			RenderableMaterial *subsetMaterial = material;

			if (const AssembledMesh *srcCompound = subsetMesh->GetCompound())
			{
				AssembledMesh::MeshRange srcMeshes = srcCompound->GetMeshes();
				uint4 srcSubsetIdx = (uint4) (std::find(srcMeshes.Begin, srcMeshes.End, subsetMesh) - srcMeshes.Begin);

				if (srcSubsetIdx < Size4(srcMeshes))
					if (beg::MaterialConfig *subsetConfig = srcCompound->GetMaterials()[srcSubsetIdx])
					{
						lean::resource_ptr<beg::Material> adaptedSubsetMaterial = beg::CreateMaterial(*material->GetMaterial());
						adaptedSubsetMaterial->SetConfigurations(&subsetConfig, 1);
						subsetMaterial = materialCache.GetMaterial(adaptedSubsetMaterial);
					}
			}
			
			mesh.SetMeshWithMaterial(subsetIdx, nullptr, subsetMaterial);
		}
}

// Adds all meshes in the given assembled mesh to the given renderable mesh using the given effect.
lean::resource_ptr<RenderableMesh, lean::critical_ref> ToRenderableMesh(const AssembledMesh &src, const beGraphics::Effect *effect,
																		RenderableMaterial *material, RenderableMaterialCache &materialCache,
																		bool bStoreSource)
{
	lean::resource_ptr<RenderableMesh> mesh = new_resource RenderableMesh(
		src.GetMeshes().Begin, src.GetMeshes().End,
		nullptr, nullptr,
		src.GetLODs().Begin, src.GetLODs().End,
		(bStoreSource) ? &src : nullptr );

	FillRenderableMesh(*mesh, material, materialCache);

	return mesh.transfer();
}

// Fills in missing materials using the given material.
void CacheMaterials(const RenderableMesh &mesh, beg::MaterialCache &materialCache)
{
	RenderableMesh::MeshRange meshes = mesh.GetMeshes();
	RenderableMesh::MaterialRange materials = mesh.GetMaterials();

	// Fill in missing material names
	for (; materials; ++materials, ++meshes)
		if (!materials[0]->GetMaterial()->GetCache())
			materialCache.SetName(
					materials[0]->GetMaterial(),
					materialCache.GetUniqueName( meshes[0]->GetName() )
				);
}

// Transfers materials from one renderable mesh to another, matching subset mesh names.
void TransferMaterials(const RenderableMesh &source, RenderableMesh &dest)
{
	RenderableMesh::MeshRange meshes = dest.GetMeshes();
	RenderableMesh::MeshRange srcMeshes = source.GetMeshes();
	RenderableMesh::MaterialRange srcMaterials = source.GetMaterials();

	for (uint4 subsetIdx = 0, subsetCount = Size4(meshes); subsetIdx < subsetCount; ++subsetIdx)
	{
		const Mesh &mesh = *meshes[subsetIdx];

		for (uint4 srcSubsetIdx = 0, srcSubsetCount = Size4(srcMeshes); srcSubsetIdx < srcSubsetCount; ++srcSubsetIdx)
			if (srcMeshes[srcSubsetIdx]->GetName() == mesh.GetName())
				dest.SetMeshWithMaterial(subsetIdx, nullptr, srcMaterials[srcSubsetIdx]);
	}
}

} // namespace