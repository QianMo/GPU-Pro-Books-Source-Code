/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beBoundMaterial.h"

namespace beScene
{

/// Technique.
struct GenericBoundMaterial::InternalTechnique
{
	const beg::MaterialTechnique *technique;
	lean::resource_ptr<EffectDriver> driver;

	/// Constructor.
	InternalTechnique(const beGraphics::MaterialTechnique *technique,
		EffectDriver *driver)
			: technique(technique),
			driver(driver) { }
};

LEAN_LAYOUT_COMPATIBLE(GenericBoundMaterialTechnique, Technique, GenericBoundMaterial::InternalTechnique, technique);
LEAN_LAYOUT_COMPATIBLE(GenericBoundMaterialTechnique, EffectDriver, GenericBoundMaterial::InternalTechnique, driver);
LEAN_SIZE_COMPATIBLE(GenericBoundMaterialTechnique, GenericBoundMaterial::InternalTechnique);

namespace
{

/// Loads techniques from the given material.
GenericBoundMaterial::techniques_t LoadTechniques(beg::Material &material, GenericEffectDriverCache &driverCache)
{
	GenericBoundMaterial::techniques_t techniques;

	const uint4 techniqueCount = material.GetTechniqueCount();
	techniques.reset(techniqueCount);

	for (uint4 techniqueIdx = 0; techniqueIdx < techniqueCount; ++techniqueIdx)
	{
		const beg::MaterialTechnique *technique = material.GetTechnique(techniqueIdx);
		EffectDriver *effectDriver = driverCache.GetEffectBinder(*technique->Technique);
		techniques.emplace_back(technique, effectDriver);
	}

	return techniques;
}

} // namespace

// Constructor.
GenericBoundMaterial::GenericBoundMaterial(beg::Material *material, GenericEffectDriverCache &driverCache)
	: m_material( LEAN_ASSERT_NOT_NULL(material) ),
	m_techniques( LoadTechniques(*material, driverCache) )
{
}

// Destructor.
GenericBoundMaterial::~GenericBoundMaterial()
{
}

// Gets the range of techniques.
GenericBoundMaterial::TechniqueRange GenericBoundMaterial::GetTechniques() const
{
	// NOTE: RenderableMaterialTechnique and RenderableMaterial::TechniqueRange asserted layout-compatible
	return beCore::MakeRangeN(
			reinterpret_cast<const GenericBoundMaterialTechnique*>(&m_techniques[0]),
			m_techniques.size()
		);
}

} // namespace
