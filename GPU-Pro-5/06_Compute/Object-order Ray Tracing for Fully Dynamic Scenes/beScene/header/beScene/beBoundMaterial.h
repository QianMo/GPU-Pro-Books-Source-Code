/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_BOUND_MATERIAL
#define BE_SCENE_BOUND_MATERIAL

#include "beScene.h"
#include <beCore/beShared.h>
#include <beCore/beMany.h>
#include <lean/tags/noncopyable.h>

#include <beCore/beManagedResource.h>
#include <beCore/beReflectedComponent.h>

#include <beGraphics/beMaterial.h>
#include "beEffectBinderCache.h"

#include <lean/smart/resource_ptr.h>
#include <lean/containers/dynamic_array.h>

namespace beScene
{

/// Bound material layer.
struct GenericBoundMaterialTechnique
{
	const beGraphics::MaterialTechnique *Technique;	///< Material properties.
	const EffectDriver *EffectDriver;				///< Effect driver.

	/// Constructor.
	GenericBoundMaterialTechnique(const beGraphics::MaterialTechnique *technique,
		const class EffectDriver *driver)
			: Technique(technique),
			EffectDriver(driver) { }
};

/// Material layer.
template <class Driver>
struct BoundMaterialTechnique : GenericBoundMaterialTechnique
{
	/// Constructor.
	BoundMaterialTechnique(const beGraphics::MaterialTechnique *technique,
		const Driver *driver)
			: GenericBoundMaterialTechnique(technique, driver) { }

	/// Gets the typed effect driver.
	const Driver* TypedDriver() const { return static_cast<const Driver*>(EffectDriver); }
	/// Sets the effect driver.
	void TypedDriver(const Driver *driver) { EffectDriver = driver; }

private:
	using GenericBoundMaterialTechnique::EffectDriver;
};

/// Binds an entire material to an effect driver.
class GenericBoundMaterial : public lean::nonassignable, public beCore::Resource
{
public:
	struct InternalTechnique;
	typedef lean::dynamic_array<InternalTechnique> techniques_t;

private:
	lean::resource_ptr<beGraphics::Material> m_material;
	techniques_t m_techniques;

	lean::resource_ptr<GenericBoundMaterial> m_pSuccessor;

public:
	/// Constructor.
	BE_SCENE_API GenericBoundMaterial(beGraphics::Material *material, GenericEffectDriverCache &driverCache);
	/// Destructor.
	BE_SCENE_API virtual ~GenericBoundMaterial();

	typedef beCore::Range<const GenericBoundMaterialTechnique*> TechniqueRange;
	/// Gets the range of techniques.
	BE_SCENE_API TechniqueRange GetTechniques() const;

	/// Gets the material.
	LEAN_INLINE beGraphics::Material* GetMaterial() { return m_material; } 
	/// Gets the material.
	LEAN_INLINE const beGraphics::Material* GetMaterial() const { return m_material; } 

	/// Sets the successor.
	LEAN_INLINE void SetSuccessor(GenericBoundMaterial *pSuccessor) { m_pSuccessor = pSuccessor; }
	/// Gets the successor.
	LEAN_INLINE GenericBoundMaterial* GetSuccessor() const { return m_pSuccessor; }

	/// Gets a pointer to the reflected component interface.
	LEAN_INLINE friend beCore::ReflectedComponent* Reflect(GenericBoundMaterial *pReflected) { return (pReflected) ? pReflected->m_material : nullptr; }
	/// Gets a pointer to the reflected component interface.
	LEAN_INLINE friend const beCore::ReflectedComponent* Reflect(const GenericBoundMaterial *pReflected) { return (pReflected) ? pReflected->m_material : nullptr; }
};

/// Binds an entire material to an effect driver.
template <class Driver, class Derived>
class BoundMaterial : public GenericBoundMaterial
{
public:
	/// Constructor.
	LEAN_INLINE BoundMaterial(beGraphics::Material *material, EffectDriverCache<Driver> &driverCache)
		: GenericBoundMaterial(material, driverCache) { }

	/// Material technique.
	typedef BoundMaterialTechnique<Driver> Technique;
	/// Range of material techniques.
	typedef beCore::Range<const Technique*> TechniqueRange;
	/// Gets the range of techniques.
	LEAN_INLINE TechniqueRange GetTechniques() const
	{
		return lean::static_range_cast<TechniqueRange>( GenericBoundMaterial::GetTechniques() );
	}

	/// Sets the successor.
	LEAN_INLINE void SetSuccessor(Derived *pSuccessor) { GenericBoundMaterial::SetSuccessor(pSuccessor); }
	/// Gets the successor.
	LEAN_INLINE Derived* GetSuccessor() const { return static_cast<Derived*>( GenericBoundMaterial::GetSuccessor() ); }
};

} // namespace

#endif