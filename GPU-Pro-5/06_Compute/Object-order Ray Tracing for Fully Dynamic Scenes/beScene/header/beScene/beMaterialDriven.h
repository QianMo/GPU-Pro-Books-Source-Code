/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_MATERIAL_DRIVEN
#define BE_SCENE_MATERIAL_DRIVEN

#include "beScene.h"
#include <beGraphics/beMaterial.h>
#include <lean/smart/resource_ptr.h>

namespace beScene
{

/// Material-driven base.
class MaterialDriven
{
protected:
	lean::resource_ptr<beGraphics::Material> m_pMaterial;	///< Material.

	MaterialDriven& operator =(const MaterialDriven&) { return *this; }
	~MaterialDriven() { }

public:
	/// Constructor.
	MaterialDriven(beGraphics::Material *pMaterial)
		: m_pMaterial(pMaterial) { }

	/// Sets the material of this renderable object.
//	virtual void SetMaterial(Material *pMaterial) = 0; // Omit V-Table

	/// Gets the material of this renderable object.
	LEAN_INLINE beGraphics::Material* GetMaterial() const { return m_pMaterial; };
};

} // namespace

#endif