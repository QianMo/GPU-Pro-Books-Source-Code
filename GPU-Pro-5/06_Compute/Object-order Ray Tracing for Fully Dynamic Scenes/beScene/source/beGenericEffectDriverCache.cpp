/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beGenericEffectDriverCache.h"

#include <beGraphics/Any/beEffect.h>
#include <beGraphics/Any/beEffectsAPI.h>

#include <unordered_map>

namespace beScene
{

/// Implementation.
class GenericDefaultEffectDriverCache::M
{
public:
	typedef std::unordered_map< ID3DX11EffectTechnique*, lean::resource_ptr<EffectDriver> > effect_drivers_t;
	effect_drivers_t effectDrivers;
};

// Constructor.
GenericDefaultEffectDriverCache::GenericDefaultEffectDriverCache()
	: m( new M() )
{
}

// Destructor.
GenericDefaultEffectDriverCache::~GenericDefaultEffectDriverCache()
{
}

// Gets an effect binder from the given effect.
EffectDriver* GenericDefaultEffectDriverCache::GetEffectBinder(const beGraphics::Technique &technique, uint4 flags)
{
	LEAN_PIMPL();

	beg::api::EffectTechnique *techniqueDX = ToImpl(technique);

	M::effect_drivers_t::const_iterator it = m.effectDrivers.find(techniqueDX);

	if (it == m.effectDrivers.end())
		it = m.effectDrivers.insert(std::make_pair( techniqueDX, CreateEffectBinder(technique, flags) )).first;

	return it->second;
}

} // namespace