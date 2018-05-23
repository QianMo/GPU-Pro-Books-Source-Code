/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_ASSEMBLED_MESH
#define BE_SCENE_ASSEMBLED_MESH

#include "beScene.h"
#include "beMeshCompound.h"
#include <beCore/beManagedResource.h>

namespace beGraphics
{
	class MaterialConfig;
}

namespace beScene
{

class MeshCache;

/// Assembled mesh.
class AssembledMesh : public MeshCompound<beGraphics::MaterialConfig>,
	public beCore::ManagedResource<MeshCache>, public beCore::HotResource<AssembledMesh>
{
public:
	/// Empty compound constructor.
	LEAN_INLINE AssembledMesh() { }
	/// Full compound constructor. Consecutive levels of detail have to reference consecutive subset ranges.
	LEAN_INLINE AssembledMesh(Mesh *const *meshBegin, Mesh *const *meshEnd,
			beGraphics::MaterialConfig *const *materialBegin, beGraphics::MaterialConfig *const *materialEnd,
			const LOD *lodBegin, const LOD *lodEnd)
		: MeshCompound<beGraphics::MaterialConfig>(meshBegin, meshEnd, materialBegin, materialEnd, lodBegin, lodEnd) { }
	/// Copies the given mesh compound.
	LEAN_INLINE AssembledMesh(const AssembledMesh &right)
		: MeshCompound<beGraphics::MaterialConfig>(right.GetMeshes().Begin, right.GetMeshes().End,
			right.GetMaterials().Begin, right.GetMaterials().End,
			right.GetLODs().Begin, right.GetLODs().End) { }

	/// Gets the type.
	BE_SCENE_API static const beCore::ComponentType* GetComponentType();
	/// Gets the type.
	BE_SCENE_API const beCore::ComponentType* GetType() const;
};

} // namespace

#endif