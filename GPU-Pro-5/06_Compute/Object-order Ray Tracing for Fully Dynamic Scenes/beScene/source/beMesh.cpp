/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beMesh.h"
#include "beScene/beMeshCompound.h"

namespace beScene
{

// Computes the combined bounds of the given mesh compound.
beMath::faab3 ComputeBounds(const Mesh *const *meshes, uint4 meshCount)
{
	beMath::faab3 result(beMath::faab3::invalid);
	const Mesh *const *meshesEnd = meshes + meshCount;

	while (meshes != meshesEnd)
	{
		const beMath::faab3 &box = (*meshes)->GetBounds();

		result.min = min_cw(box.min, result.min);
		result.max = max_cw(box.max, result.max);
		
		++meshes;
	}

	return result;
}

// Constructor.
Mesh::Mesh(const utf8_ntri& name, AssembledMesh *pCompound)
	: m_name(name.to<utf8_string>()),
	m_pCompound(pCompound)
{
}

// Constructor.
Mesh::Mesh(const utf8_ntri& name, const beMath::faab3& bounds, AssembledMesh *pCompound)
	: m_name(name.to<utf8_string>()),
	m_pCompound(pCompound),
	m_bounds(bounds)
{
}

// Destructor.
Mesh::~Mesh()
{
}

} // namespace