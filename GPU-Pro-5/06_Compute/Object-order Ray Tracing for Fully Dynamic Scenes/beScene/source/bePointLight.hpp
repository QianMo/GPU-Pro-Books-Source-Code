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
const beCore::ReflectionProperty LightInternals<PointLightController>::Properties[] =
{
	beCore::MakeReflectionProperty<float[4]>("color", beCore::Widget::Color)
		.set_setter( BE_CORE_PROPERTY_SETTER_UNION(&PointLightController::SetColor, float) )
		.set_getter( BE_CORE_PROPERTY_GETTER_UNION(&PointLightController::GetColor, float) ),
	beCore::MakeReflectionProperty<float[4]>("indirect color", beCore::Widget::Color)
		.set_setter( BE_CORE_PROPERTY_SETTER_UNION(&PointLightController::SetIndirectColor, float) )
		.set_getter( BE_CORE_PROPERTY_GETTER_UNION(&PointLightController::GetIndirectColor, float) ),
	beCore::MakeReflectionProperty<float>("attenuation", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&PointLightController::SetAttenuation) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&PointLightController::GetAttenuation) ),
	beCore::MakeReflectionProperty<float>("attenuation offset", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&PointLightController::SetAttenuationOffset) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&PointLightController::GetAttenuationOffset) ),
	beCore::MakeReflectionProperty<float>("range", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&PointLightController::SetRange) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&PointLightController::GetRange) ),
	beCore::MakeReflectionProperty<bool>("shadow", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&PointLightController::EnableShadow) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&PointLightController::IsShadowEnabled) ),
	beCore::MakeReflectionProperty<uint4>("shadow resolution", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&PointLightController::SetShadowResolution) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&PointLightController::GetShadowResolution) )
};

template <>
const beCore::ComponentType LightInternals<PointLightController>::ControllerType = { "PointLightController" };
template <>
const beCore::ComponentType LightInternals< LightControllers<PointLightController> >::ControllerType = { "PointLightControllers" };

template <>
const uint4 LightInternals< PointLightController >::LightTypeID = GetLightTypes().GetID("PointLight");
template <>
utf8_string LightInternals< PointLightController >::DefaultEffectFile = "Lights/PointLight.fx";

template <>
bem::fsphere3 LightInternals< PointLightController >::GetDefaultBounds()
{
	return bem::fsphere3(bem::fvec3(), 1.0f);
}

template <>
struct LightInternals<PointLightController>::Constants
{
	bem::fmat4 Transformation;		///< Location.

	bem::fvec4 Color;				///< Light color.
	bem::fvec4 IndirectColor;		///< Indirect (ambient) color.

	float Attenuation;				///< Light attenuation.
	float AttenuationOffset;		///< Light attenuation offset.
	float Range;					///< Light range.
	float _1;

	Constants()
		: Color(1.0f),
		IndirectColor(1.0f),
		Attenuation(1.0f),
		AttenuationOffset(1.0f),
		Range(25.0f) { }
};

template <>
struct LightInternals<PointLightController>::ShadowConstants
{
	bem::fvec2 ShadowResolution;	///< Shadow resolution.
	bem::fvec2 ShadowPixel;			///< Shadow pixel (= 1 / resolution).

	/// Shadow cube map face constants.
	struct ShadowFace
	{
		bem::fmat4 Proj;		///< Shadow projection matrix.
	};

	static const uint4 ShadowFaceCount = 6;		///< Shade map face count.
	ShadowFace ShadowFaces[ShadowFaceCount];	///< Shadow map faces.
};

template <>
struct LightInternals<PointLightController>::ShadowState : LightInternals<PointLightController>::ShadowStateBase
{
	static const uint4 ShadowFaceCount = LightInternals<PointLightController>::ShadowConstants::ShadowFaceCount;

	lean::com_ptr<PipelinePerspective> Perspective[ShadowFaceCount];

	ShadowState(const ShadowStateBase &base)
		: ShadowStateBase(base) { }
};

namespace PointShadows
{

/// Computes the shadow orientation.
inline beMath::fmat3 ComputeShadowOrientation(const beMath::fmat3 &orientation, uint4 index)
{
	float viewSign = (index < 3) ? 1.0f : -1.0f;

	return beMath::mat_transform3(
		orientation[(2 + index) % 2] * viewSign,
		orientation[(1 + index) % 2],
		orientation[(0 + index) % 2] * viewSign);
}

/// Computes the three shadow matrices.
inline void ComputeShadowMatrices(beMath::fmat4 &view, beMath::fmat4 &proj, beMath::fmat4 &viewProj, float &nearPlane, float &farPlane, 
	const beMath::fmat3 &orientation, const beMath::fvec3 &position, float range)
{
	const float Eps = 0.1f;

	view = beMath::mat_view(position, orientation[2], orientation[1], orientation[0]);
	nearPlane = Eps;
	farPlane = nearPlane + range;
	proj = beMath::mat_proj(beMath::pi<float>::quarter, 1.0f, nearPlane, farPlane);
	viewProj = mul(view, proj);
}

} // namespace

/// Adds a shadow.
uint4 AddShadow(LightInternals<PointLightController>::shadow_state_t &shadows,
				const LightInternals<PointLightController>::ShadowStateBase &shadowStateBase,
				const LightInternals<PointLightController>::Configuration &state,
				PipelinePerspective &perspective, PerspectivePool &perspectivePool)
{
	const uint4 FaceCount = LightInternals<PointLightController>::ShadowState::ShadowFaceCount;
	
	const beGraphics::TextureTargetDesc targetDesc(
			state.ShadowResolution, state.ShadowResolution,
			1,
			beGraphics::Format::R16F,
			beGraphics::SampleDesc(),
			FaceCount
		);
	lean::resource_ptr<Pipe> splitPipe = perspectivePool.GetPipePool()->GetPipe(targetDesc);

	uint4 shadowOffset = (uint4) shadows.size();
	LightInternals<PointLightController>::ShadowState &shadowState = new_emplace(shadows)
		LightInternals<PointLightController>::ShadowState(shadowStateBase);

	for (uint4 i = 0; i < FaceCount; ++i)
		shadowState.Perspective[i] = perspectivePool.GetPerspective(splitPipe, nullptr, state.ShadowStageMask);

	return shadowOffset;
}

/// Prepares the shadow data for rendering.
void PrepareShadow(const LightInternals<PointLightController>::Configuration &state,
				  const LightInternals<PointLightController>::Constants &constants,
				  LightInternals<PointLightController>::ShadowConstants &shadowConstants,
				  const LightInternals<PointLightController>::ShadowState &shadowState,
				  PipelinePerspective &perspective)
{
	using namespace PointShadows;
	const uint4 FaceCount = LightInternals<PointLightController>::ShadowConstants::ShadowFaceCount;

	const beMath::fvec3 lightPos(constants.Transformation[3]);
	const beMath::fmat3 lightOrientation(constants.Transformation);

	const PerspectiveDesc &camPerspectiveDesc = perspective.GetDesc();
	const beMath::fmat3 camOrientation = mat_transform3(camPerspectiveDesc.CamLook, camPerspectiveDesc.CamUp, camPerspectiveDesc.CamRight);

	shadowConstants.ShadowResolution = (float) state.ShadowResolution;
	shadowConstants.ShadowPixel = 1.0f / shadowConstants.ShadowResolution;

	for (uint4 i = 0; i < FaceCount; ++i)
	{
		LightInternals<PointLightController>::ShadowConstants::ShadowFace &shadowFace = shadowConstants.ShadowFaces[i];

		beMath::fmat3 faceOrientation = ComputeShadowOrientation(lightOrientation, i);

		beMath::fmat4 faceView, faceProj, faceViewProj;
		float faceMin, faceMax;
		ComputeShadowMatrices(
				faceView, faceProj, faceViewProj, faceMin, faceMax,
				faceOrientation, lightPos, constants.Range
			);

		shadowFace.Proj = faceViewProj;

		PerspectiveDesc facePerspectiveDesc(
					lightPos,
					faceOrientation[0],
					faceOrientation[1],
					faceOrientation[2],
					faceView,
					faceProj,
					faceMin,
					faceMax,
					camPerspectiveDesc.Flipped,
					camPerspectiveDesc.Time,
					camPerspectiveDesc.TimeStep,
					i
				);
		shadowState.Perspective[i]->SetDesc(facePerspectiveDesc);

		perspective.AddPerspective(shadowState.Perspective[i]);
	}
}

// Gets shadow maps.
beGraphics::TextureViewHandle GatherShadow(const LightInternals<PointLightController>::ShadowState &shadowState)
{
	const beg::TextureTarget *pTarget = (shadowState.Perspective[0]) ? shadowState.Perspective[0]->GetPipe()->GetAnyTarget("SceneShadowTarget") : nullptr;
	return beg::Any::TextureViewHandle( (pTarget) ? pTarget->GetTexture() : nullptr );
}

template LightControllers<PointLightController>;
template LightControllerBase<PointLightController>;
template void InstantiateLightControllerFunctions<PointLightController>();

} // namespace
