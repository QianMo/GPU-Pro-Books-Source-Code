/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beAssembledMesh.h"
#include <beGraphics/beMaterialConfig.h>

namespace beScene
{
	extern const beCore::ComponentType AssembledMeshType = { "AssembledMesh" };

	template MeshCompound<beg::MaterialConfig>;

	/// Gets the type.
	const beCore::ComponentType* AssembledMesh::GetComponentType()
	{
		return &AssembledMeshType;
	}
	// Gets the type.
	const beCore::ComponentType* AssembledMesh::GetType() const
	{
		return &AssembledMeshType;
	}

} // namespace

#include "beMeshCompound.cpp"