/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beScene/beLightControllers.h"

#include <beCore/beReflectionProperties.h>

#include <beCore/beIdentifiers.h>
#include "beScene/beLight.h"

#include "beScene/bePipePool.h"

#include <beMath/beProjection.h>
#include <beMath/beMatrix.h>

namespace beScene
{

template <>
const beCore::ReflectionProperty LightInternals<SpotLightController>::Properties[] =
{
	beCore::MakeReflectionProperty<float[4]>("color", beCore::Widget::Color)
		.set_setter( BE_CORE_PROPERTY_SETTER_UNION(&SpotLightController::SetColor, float) )
		.set_getter( BE_CORE_PROPERTY_GETTER_UNION(&SpotLightController::GetColor, float) ),
	beCore::MakeReflectionProperty<float[4]>("indirect color", beCore::Widget::Color)
		.set_setter( BE_CORE_PROPERTY_SETTER_UNION(&SpotLightController::SetIndirectColor, float) )
		.set_getter( BE_CORE_PROPERTY_GETTER_UNION(&SpotLightController::GetIndirectColor, float) ),
	beCore::MakeReflectionProperty<float>("attenuation", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&SpotLightController::SetAttenuation) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&SpotLightController::GetAttenuation) ),
	beCore::MakeReflectionProperty<float>("attenuation offset", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&SpotLightController::SetAttenuationOffset) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&SpotLightController::GetAttenuationOffset) ),
	beCore::MakeReflectionProperty<float>("range", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&SpotLightController::SetRange) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&SpotLightController::GetRange) ),
	beCore::MakeReflectionProperty<float>("inner angle", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&SpotLightController::SetInnerAngle) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&SpotLightController::GetInnerAngle) ),
	beCore::MakeReflectionProperty<float>("outer angle", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&SpotLightController::SetOuterAngle) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&SpotLightController::GetOuterAngle) ),
	beCore::MakeReflectionProperty<bool>("shadow", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&SpotLightController::EnableShadow) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&SpotLightController::IsShadowEnabled) ),
	beCore::MakeReflectionProperty<uint4>("shadow resolution", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&SpotLightController::SetShadowResolution) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&SpotLightController::GetShadowResolution) )
};

template <>
const beCore::ComponentType LightInternals<SpotLightController>::ControllerType = { "SpotLightController" };
template <>
const beCore::ComponentType LightInternals< LightControllers<SpotLightController> >::ControllerType = { "SpotLightControllers" };

template <>
const uint4 LightInternals< SpotLightController >::LightTypeID = GetLightTypes().GetID("SpotLight");
template <>
utf8_string LightInternals< SpotLightController >::DefaultEffectFile = "Lights/SpotLight.fx";

template <>
bem::fsphere3 LightInternals< SpotLightController >::GetDefaultBounds()
{
	return bem::fsphere3(bem::vec(0.0f, 0.0f, 0.5f), sqrt(0.75f));
}

const float SpotLightDefaultInnerAngle = 0.6f;
const float SpotLightDefaultOuterAngle = 1.2f;

template <>
struct LightInternals<SpotLightController>::AuxConfiguration
{
	float InnerAngle;			///< Inner spot cone angle.
	float OuterAngle;			///< Outer spot cone angle.

	AuxConfiguration()
		: InnerAngle(SpotLightDefaultInnerAngle),
		OuterAngle(SpotLightDefaultOuterAngle) { }
};

template <>
struct LightInternals<SpotLightController>::Constants
{
	bem::fmat4 Transformation;		///< Location.

	bem::fvec4 Color;				///< Light color.
	bem::fvec4 IndirectColor;		///< Indirect (ambient) color.

	float Attenuation;				///< Light attenuation.
	float AttenuationOffset;		///< Light attenuation offset.
	float Range;					///< Light range.
	float _1;

	float CosInnerAngle;			///< Cosine of inner spot cone angle.
	float CosOuterAngle;			///< Cosine of outer spot cone angle.
	float SinInnerAngle;			///< Sine of inner spot cone angle.
	float SinOuterAngle;			///< Sine of outer spot cone angle.

	Constants()
		: Color(1.0f),
		IndirectColor(1.0f),
		Attenuation(1.0f),
		AttenuationOffset(1.0f),
		Range(25.0f),
		CosInnerAngle(cos(SpotLightDefaultInnerAngle)),
		CosOuterAngle(cos(SpotLightDefaultOuterAngle)),
		SinInnerAngle(sin(SpotLightDefaultInnerAngle)),
		SinOuterAngle(sin(SpotLightDefaultOuterAngle)) { }
};

template <>
struct LightInternals<SpotLightController>::ShadowConstants
{
	bem::fvec2 ShadowResolution;	///< Shadow resolution.
	bem::fvec2 ShadowPixel;			///< Shadow pixel (= 1 / resolution).
	bem::fmat4 ShadowProj;			///< Shadow projection matrix.
};

template <>
struct LightInternals<SpotLightController>::ShadowState : LightInternals<SpotLightController>::ShadowStateBase
{
	lean::com_ptr<PipelinePerspective> Perspective;

	ShadowState(const ShadowStateBase &base)
		: ShadowStateBase(base) { }
};

namespace SpotShadows
{

/// Computes the three shadow matrices.
inline void ComputeShadowMatrices(beMath::fmat4 &view, beMath::fmat4 &proj, beMath::fmat4 &viewProj, float &nearPlane, float &farPlane, 
	const beMath::fmat3 &orientation, const beMath::fvec3 &position, float angle, float range)
{
	const float Eps = 0.1f;

	view = beMath::mat_view(position, orientation[2], orientation[1], orientation[0]);
	nearPlane = Eps;
	farPlane = nearPlane + range;
	proj = beMath::mat_proj(angle, 1.0f, nearPlane, farPlane);
	viewProj = mul(view, proj);
}

} // namespace

/// Adds a shadow.
uint4 AddShadow(LightInternals<SpotLightController>::shadow_state_t &shadows,
				const LightInternals<SpotLightController>::ShadowStateBase &shadowStateBase,
				const LightInternals<SpotLightController>::Configuration &state,
				PipelinePerspective &perspective, PerspectivePool &perspectivePool)
{
	const beGraphics::TextureTargetDesc targetDesc(
			state.ShadowResolution, state.ShadowResolution,
			1,
			beGraphics::Format::R16F,
			beGraphics::SampleDesc()
		);
	lean::resource_ptr<Pipe> splitPipe = perspectivePool.GetPipePool()->GetPipe(targetDesc);

	uint4 shadowOffset = (uint4) shadows.size();
	LightInternals<SpotLightController>::ShadowState &shadowState = new_emplace(shadows)
		LightInternals<SpotLightController>::ShadowState(shadowStateBase);

	shadowState.Perspective = perspectivePool.GetPerspective(splitPipe, nullptr, state.ShadowStageMask);

	return shadowOffset;
}

/// Prepares the shadow data for rendering.
void PrepareShadow(const LightInternals<SpotLightController>::Configuration &state,
				  const LightInternals<SpotLightController>::Constants &constants,
				  LightInternals<SpotLightController>::ShadowConstants &shadowConstants,
				  const LightInternals<SpotLightController>::ShadowState &shadowState,
				  PipelinePerspective &perspective)
{
	using namespace SpotShadows;
	
	const beMath::fvec3 lightPos(constants.Transformation[3]);
	const beMath::fmat3 lightOrientation(constants.Transformation);

	const PerspectiveDesc &camPerspectiveDesc = perspective.GetDesc();
	const beMath::fmat3 camOrientation = mat_transform3(camPerspectiveDesc.CamLook, camPerspectiveDesc.CamUp, camPerspectiveDesc.CamRight);

	shadowConstants.ShadowResolution = (float) state.ShadowResolution;
	shadowConstants.ShadowPixel = 1.0f / shadowConstants.ShadowResolution;

	beMath::fmat4 faceView, faceProj, faceViewProj;
	float faceMin, faceMax;
	ComputeShadowMatrices(
			faceView, faceProj, faceViewProj, faceMin, faceMax,
			lightOrientation, lightPos, state.OuterAngle, constants.Range
		);

	shadowConstants.ShadowProj = faceViewProj;

	PerspectiveDesc facePerspectiveDesc(
				lightPos,
				lightOrientation[0],
				lightOrientation[1],
				lightOrientation[2],
				faceView,
				faceProj,
				faceMin,
				faceMax,
				camPerspectiveDesc.Flipped,
				camPerspectiveDesc.Time,
				camPerspectiveDesc.TimeStep
			);
	shadowState.Perspective->SetDesc(facePerspectiveDesc);

	perspective.AddPerspective(shadowState.Perspective);
}

// Gets shadow maps.
beGraphics::TextureViewHandle GatherShadow(const LightInternals<SpotLightController>::ShadowState &shadowState)
{
	const beg::TextureTarget *pTarget = (shadowState.Perspective) ? shadowState.Perspective->GetPipe()->GetAnyTarget("SceneShadowTarget") : nullptr;
	return beg::Any::TextureViewHandle( (pTarget) ? pTarget->GetTexture() : nullptr );
}

template LightControllers<SpotLightController>;
template LightControllerBase<SpotLightController>;
template void InstantiateLightControllerFunctions<SpotLightController>();

} // namespace
