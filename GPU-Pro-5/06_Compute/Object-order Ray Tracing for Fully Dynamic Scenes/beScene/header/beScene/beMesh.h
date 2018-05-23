/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_MESH
#define BE_SCENE_MESH

#include "beScene.h"
#include <beCore/beShared.h>
#include <beMath/beAABDef.h>
#include <beGraphics/beGraphics.h>
#include <string>

namespace beScene
{

class AssembledMesh;

/// Mesh base.
class Mesh : public beCore::Resource, public beGraphics::Implementation
{
protected:
	lean::utf8_string m_name;

	AssembledMesh *m_pCompound;		///< Compound.

	beMath::faab3 m_bounds;			///< Bounding box.

	LEAN_INLINE Mesh& operator =(const Mesh&) { return *this; }

public:
	/// Constructor.
	Mesh(const utf8_ntri& name, AssembledMesh *pCompound);
	/// Constructor.
	Mesh(const utf8_ntri& name, const beMath::faab3& bounds, AssembledMesh *pCompound);
	/// Destructor.
	virtual ~Mesh();

	/// Gets the bounding box.
	LEAN_INLINE const beMath::faab3& GetBounds() const { return m_bounds; } 

	/// Gets the name.
	LEAN_INLINE const lean::utf8_string& GetName() const { return m_name; }

	/// Gets the mesh compound.
	LEAN_INLINE AssembledMesh* GetCompound() const { return m_pCompound; }
};

/// Computes the combined bounds of the given mesh compound.
BE_SCENE_API beMath::faab3 ComputeBounds(const Mesh *const *meshes, uint4 meshCount);

} // namespace

#endif