/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_LIGHT_EFFECT_BINDER
#define BE_SCENE_LIGHT_EFFECT_BINDER

#include "beScene.h"
#include "beEffectBinder.h"
#include "beLightEffectData.h"
#include <beGraphics/Any/beEffect.h>
#include <beGraphics/Any/beEffectsAPI.h>
#include <lean/smart/resource_ptr.h>
#include <vector>

namespace beScene
{

// Prototypes
struct LightJob;
class RenderingPipeline;

/// Lighting effect binder.
class LightEffectBinder : public EffectBinder
{
public:
	struct Pass;
	typedef std::vector<Pass> pass_vector;

private:
	const beGraphics::Any::Technique m_technique;

	beGraphics::API::EffectConstantBuffer *m_pLightData;
	beGraphics::API::EffectConstantBuffer *m_pShadowData;
	beGraphics::API::EffectShaderResource *m_pShadowMaps;

	pass_vector m_passes;

public:
	/// Constructor.
	BE_SCENE_API LightEffectBinder(const beGraphics::Any::Technique &technique, uint4 passID = static_cast<uint4>(-1));
	/// Destructor.
	BE_SCENE_API ~LightEffectBinder();

	/// Applies the n-th step of the given pass.
	BE_SCENE_API bool Apply(uint4 &nextPassID, const LightEffectData &light, beGraphics::Any::API::DeviceContext *pContext) const;

	/// Gets the technique.
	LEAN_INLINE const beGraphics::Technique& GetTechnique() const { return m_technique; }
	/// Gets the effect.
	LEAN_INLINE const beGraphics::Effect& GetEffect() const { return *m_technique.GetEffect(); }
};

} // namespace

#endif