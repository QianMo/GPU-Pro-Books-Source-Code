/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PIPE_EFFECT_BINDER
#define BE_SCENE_PIPE_EFFECT_BINDER

#include "beScene.h"
#include "beEffectBinder.h"
#include <beGraphics/Any/beEffect.h>
#include <beGraphics/Any/beEffectsAPI.h>
#include <beGraphics/Any/beStateManager.h>
#include <lean/smart/resource_ptr.h>
#include <vector>

namespace beScene
{

namespace DX11
{
	// Prototypes
	class Pipe;
}

/// Pipe effect binder flags enumeration.
struct PipeEffectBinderFlags
{
	/// Enumeration.
	enum T
	{
		NoDefaultMS = 0x1	///< Default multisamplg to off.
	};
	LEAN_MAKE_ENUM_STRUCT(PipeEffectBinderFlags)
};

/// Pipe effect binder.
class PipeEffectBinder : public EffectBinder
{
public:
	struct Pass;
	typedef std::vector<Pass> pass_vector;

	struct Target;
	typedef std::vector<Target> target_vector;

	typedef std::vector<bool> used_bitset;

private:
	const beGraphics::Any::Technique m_technique;

	target_vector m_targets;

	pass_vector m_passes;

	uint4 m_targetsUsedPitch;
	used_bitset m_targetsUsed;

	beGraphics::Any::API::EffectVector *m_pResolution;
	beGraphics::Any::API::EffectScalar *m_pMultisampling;

	beGraphics::Any::API::EffectVector *m_pDestinationScaling;
	beGraphics::Any::API::EffectVector *m_pDestinationResolution;
	beGraphics::Any::API::EffectScalar *m_pDestinationMultisampling;

public:
	/// Constructor.
	BE_SCENE_API PipeEffectBinder(const beGraphics::Any::Technique &technique, uint4 flags = 0, uint4 passID = static_cast<uint4>(-1));
	/// Destructor.
	BE_SCENE_API ~PipeEffectBinder();

	/// Applies the n-th step of the given pass.
	BE_SCENE_API bool Apply(uint4 &nextPassID, DX11::Pipe *pPipe, uint4 outputIndex, const void *pObject,
		beGraphics::Any::StateManager& stateManager, beGraphics::Any::API::DeviceContext *pContext) const;

	/// Gets the technique.
	LEAN_INLINE const beGraphics::Technique& GetTechnique() const { return m_technique; }
	/// Gets the effect.
	LEAN_INLINE const beGraphics::Effect& GetEffect() const { return *m_technique.GetEffect(); }
};

} // namespace

#endif