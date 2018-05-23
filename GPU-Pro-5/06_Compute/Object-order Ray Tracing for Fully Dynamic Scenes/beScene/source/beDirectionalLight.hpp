/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beScene/beLightControllers.h"

#include <beCore/beReflectionProperties.h>

#include <beCore/beIdentifiers.h>
#include "beScene/beLight.h"

#include "beScene/bePipePool.h"
#include "beScene/bePipe.h"
#include <beGraphics/Any/beTextureTargetPool.h>

#include <beMath/beProjection.h>
#include <beMath/beMatrix.h>

namespace beScene
{

template <>
const beCore::ReflectionProperty LightInternals<DirectionalLightController>::Properties[] =
{
	beCore::MakeReflectionProperty<float[4]>("color", beCore::Widget::Color)
		.set_setter( BE_CORE_PROPERTY_SETTER_UNION(&DirectionalLightController::SetColor, float) )
		.set_getter( BE_CORE_PROPERTY_GETTER_UNION(&DirectionalLightController::GetColor, float) ),
	beCore::MakeReflectionProperty<float[4]>("indirect color", beCore::Widget::Color)
		.set_setter( BE_CORE_PROPERTY_SETTER_UNION(&DirectionalLightController::SetIndirectColor, float) )
		.set_getter( BE_CORE_PROPERTY_GETTER_UNION(&DirectionalLightController::GetIndirectColor, float) ),
	beCore::MakeReflectionProperty<float[4]>("sky color", beCore::Widget::Color, bec::PropertyPersistence::Read) // TODO: COMPATIBILITY: Get rid of
		.set_setter( BE_CORE_PROPERTY_SETTER_UNION(&DirectionalLightController::SetIndirectColor, float) )
		.set_getter( BE_CORE_PROPERTY_GETTER_UNION(&DirectionalLightController::GetIndirectColor, float) ),
	beCore::MakeReflectionProperty<bool>("shadow", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&DirectionalLightController::EnableShadow) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&DirectionalLightController::IsShadowEnabled) ),
	beCore::MakeReflectionProperty<uint4>("shadow resolution", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&DirectionalLightController::SetShadowResolution) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&DirectionalLightController::GetShadowResolution) )
};

template <>
const beCore::ComponentType LightInternals<DirectionalLightController>::ControllerType = { "DirectionalLightController" };
template <>
const beCore::ComponentType LightInternals< LightControllers<DirectionalLightController> >::ControllerType = { "DirectionalLightControllers" };

template <>
const uint4 LightInternals< DirectionalLightController >::LightTypeID = GetLightTypes().GetID("DirectionalLight");
template <>
utf8_string LightInternals< DirectionalLightController >::DefaultEffectFile = "Lights/DirectionalLight.fx";

template <>
bem::fsphere3 LightInternals< DirectionalLightController >::GetDefaultBounds()
{
	return bem::fsphere3(bem::fvec3(), FLT_MAX * 0.5f);
}

template <>
struct LightInternals<DirectionalLightController>::Constants
{
	bem::fmat4 Transformation;		///< Location.

	bem::fvec4 Color;				///< Light color.
	bem::fvec4 IndirectColor;		///< Indirect (ambient) color.

	Constants()
		: Color(1.0f),
		IndirectColor(1.0f) { }
};

template <>
struct LightInternals<DirectionalLightController>::ShadowConstants
{
	bem::fvec2 ShadowResolution;	///< Shadow resolution.
	bem::fvec2 ShadowPixel;			///< Shadow pixel (= 1 / resolution).

	/// Shadow split constants.
	struct ShadowSplit
	{
		bem::fmat4 Proj;		///< Shadow projection matrix.
		float Near;				///< Near plane.
		float Far;				///< Far plane.
		bem::fvec2 PixelScale;	///< 1 / pixel width.
	};

	static const uint4 MaxShadowSplitCount = 4;		///< Maximum shadow split count.
	float ShadowSplitPlanes[MaxShadowSplitCount];	///< Split plane distances
	ShadowSplit ShadowSplits[MaxShadowSplitCount];	///< Shadow spits.
};

template <>
struct LightInternals<DirectionalLightController>::ShadowState : LightInternals<DirectionalLightController>::ShadowStateBase
{
	static const uint4 MaxShadowSplitCount = LightInternals<DirectionalLightController>::ShadowConstants::MaxShadowSplitCount;

	lean::com_ptr<PipelinePerspective> Perspective[MaxShadowSplitCount];

	ShadowState(const ShadowStateBase &base)
		: ShadowStateBase(base) { }
};

namespace DirectionalShadows
{

/// Computes the shadow orientation.
inline float ComputeSplitOffset(float nearPlane, float farPlane, uint4 index, uint4 splitCount)
{
	float percentage = (float) index / splitCount;

	return 0.5f * (
			nearPlane + (farPlane - nearPlane) * percentage
			+ nearPlane * pow(farPlane / nearPlane, percentage)
		);
}

/// Computes the three shadow matrices.
void ComputeShadowMatrices(beMath::fmat4 &view, beMath::fmat4 &proj, beMath::fmat4 &viewProj,
						   beMath::fvec2 &pixelScale, beMath::fvec3 &center, float &nearPlane, float &farPlane,
						   const beMath::fmat3 &camOrientation, const beMath::fvec3 &camPosition, const beMath::fmat4 &camViewProj,
						   float splitNear, float splitFar,
						   const beMath::fmat3 &splitOrientation, float range, uint4 resolution,
						   bool bOmnidirectional)
{
	center = camPosition;
	if (!bOmnidirectional)
		center += 0.5f * (splitNear + splitFar) * camOrientation[2];

	view = beMath::mat_view(center, splitOrientation[2], splitOrientation[1], splitOrientation[0]);

	beMath::fvec3 cornerPoints[8];

	if (!bOmnidirectional)
	{
		beMath::fvec3 centers[2];
		centers[0] = camPosition + splitNear * camOrientation[2];
		centers[1] = camPosition + splitFar * camOrientation[2];

		beMath::fplane3 sides[2];
		sides[0] = frustum_left(camViewProj);
		sides[1] = frustum_right(camViewProj);

		beMath::fvec3 sideCenters[4];

		for (int i = 0; i < 2; ++i)
		{
			// Project side normal to camera plane
			beMath::fvec3 toSideDir = sides[i].n() - camOrientation[2] * dot(sides[i].n(), camOrientation[2]);
			// Scale to erase side plane distance
			toSideDir /= -dot(toSideDir, sides[i].n());
		
			for (int j = 0; j < 2; ++j)
				// Project split centers to side
				sideCenters[2 * j + i] = centers[j] + toSideDir * sdist(sides[i], centers[j]);
		}

		beMath::fplane3 vertSides[2];
		vertSides[0] = frustum_top(camViewProj);
		vertSides[1] = frustum_bottom(camViewProj);

		for (int i = 0; i < 2; ++i)
		{
			// Compute vector orthogonal to both side & cam plane
			beMath::fvec3 orthoSide = cross(sides[i].n(), camOrientation[2]);
		
			for (int j = 0; j < 2; ++j)
			{
				// Scale to erase vertical side plane distance
				beMath::fvec3 toVertSide = orthoSide / -dot(orthoSide, vertSides[j].n());

				for (int k = 0; k < 2; ++k)
				{
					const beMath::fvec3 &sideCenter = sideCenters[2 * k + i];

					// Project side center to vertical side
					cornerPoints[4 * k + 2 * j + i] = sideCenter + toVertSide * sdist(vertSides[j], sideCenter);
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < 2; ++i)
			for (int j = 0; j < 2; ++j)
				for (int k = 0; k < 2; ++k)
					cornerPoints[4 * k + 2 * j + i] = camPosition
						+ (i ? -splitFar : splitFar) * camOrientation[2]
						+ (j ? -splitFar : splitFar) * camOrientation[1]
						+ (k ? -splitFar : splitFar) * camOrientation[0];
	}

	beMath::fvec3 splitSpaceMin(2.0e16f), splitSpaceMax(-2.0e16f);

	for (int i = 0; i < 8; ++i)
	{
		// Transform split corner points to split space to compute bounds
		beMath::fvec3 splitSpaceCorner = mulh(cornerPoints[i], view);
		splitSpaceMin = min_cw(splitSpaceCorner, splitSpaceMin);
		splitSpaceMax = max_cw(splitSpaceCorner, splitSpaceMax);
	}

	// Snap resolution changes
	float pixelWidth = (splitSpaceMax[0] - splitSpaceMin[0]) / resolution;
	float pixelHeight = (splitSpaceMax[1] - splitSpaceMin[1]) / resolution;

	static const float Log2 = log(2.0f);

	float snappedPixelWidth = exp( Log2 * ceil( log(pixelWidth) / Log2 ) );
	float snappedPixelHeight = exp( Log2 * ceil( log(pixelHeight) / Log2 ) );

	float snappedRangeX = snappedPixelWidth * resolution;
	float snappedRangeY = snappedPixelHeight * resolution;

	// Snap view
	center -= splitOrientation[0] * fmod( dot(center, splitOrientation[0]), snappedPixelWidth );
	center -= splitOrientation[1] * fmod( dot(center, splitOrientation[1]), snappedPixelHeight );

	view = beMath::mat_view(center, splitOrientation[2], splitOrientation[1], splitOrientation[0]);
	
	// Snap projection
	splitSpaceMin[0] = floor(splitSpaceMin[0] / snappedPixelWidth - 1.0f) * snappedPixelWidth;
	splitSpaceMin[1] = floor(splitSpaceMin[1] / snappedPixelHeight - 1.0f) * snappedPixelHeight;
	// Need EXACTLY the snapped resolution
	splitSpaceMax[0] = splitSpaceMin[0] + snappedRangeX;
	splitSpaceMax[1] = splitSpaceMin[1] + snappedRangeY;

	nearPlane = splitSpaceMin[2] - 32.0f * range;
	farPlane = splitSpaceMax[2];
	proj = beMath::mat_proj_ortho(splitSpaceMin[0], splitSpaceMin[1],
		splitSpaceMax[0], splitSpaceMax[1],
		nearPlane, farPlane);
	
	viewProj = mul(view, proj);

	pixelScale[0] = 1.0f / snappedRangeX;
	pixelScale[1] = 1.0f / snappedRangeY;
}

} // namespace

/// Adds a shadow.
uint4 AddShadow(LightInternals<DirectionalLightController>::shadow_state_t &shadows,
				const LightInternals<DirectionalLightController>::ShadowStateBase &shadowStateBase,
				const LightInternals<DirectionalLightController>::Configuration &state,
				PipelinePerspective &perspective, PerspectivePool &perspectivePool)
{
	const uint4 SplitCount = LightInternals<DirectionalLightController>::ShadowState::MaxShadowSplitCount;
	
	const beGraphics::TextureTargetDesc targetDesc(
			state.ShadowResolution, state.ShadowResolution,
			1,
			beGraphics::Format::R16F,
			beGraphics::SampleDesc(),
			SplitCount
		);
	lean::resource_ptr<Pipe> splitPipe = perspectivePool.GetPipePool()->GetPipe(targetDesc);

	uint4 shadowOffset = (uint4) shadows.size();
	LightInternals<DirectionalLightController>::ShadowState &shadowState = new_emplace(shadows)
		LightInternals<DirectionalLightController>::ShadowState(shadowStateBase);

	for (uint4 i = 0; i < SplitCount; ++i)
		shadowState.Perspective[i] = perspectivePool.GetPerspective(splitPipe, nullptr, state.ShadowStageMask);

	return shadowOffset;
}

/// Prepares the shadow data for rendering.
void PrepareShadow(const LightInternals<DirectionalLightController>::Configuration &state,
				  const LightInternals<DirectionalLightController>::Constants &constants,
				  LightInternals<DirectionalLightController>::ShadowConstants &shadowConstants,
				  const LightInternals<DirectionalLightController>::ShadowState &shadowState,
				  PipelinePerspective &perspective)
{
	using namespace DirectionalShadows;

	const uint4 SplitCount = LightInternals<DirectionalLightController>::ShadowConstants::MaxShadowSplitCount;
	// TODO TODO TODO
	const float ShadowRange = 256.0f;

	const beMath::fmat3 splitOrientation(constants.Transformation);

	const PerspectiveDesc &perspectiveDesc = perspective.GetDesc();
	const beMath::fmat3 camOrientation = mat_transform3(perspectiveDesc.CamLook, perspectiveDesc.CamUp, perspectiveDesc.CamRight);

	shadowConstants.ShadowResolution = (float) state.ShadowResolution;
	shadowConstants.ShadowPixel = 1.0f / shadowConstants.ShadowResolution;

	for (uint4 i = 0; i < SplitCount; ++i)
	{
		LightInternals<DirectionalLightController>::ShadowConstants::ShadowSplit &shadowSplit = shadowConstants.ShadowSplits[i];

		float splitNear = ComputeSplitOffset(perspectiveDesc.NearPlane, ShadowRange, i, SplitCount);
		float splitFar = ComputeSplitOffset(perspectiveDesc.NearPlane, ShadowRange, i + 1, SplitCount);

		shadowConstants.ShadowSplitPlanes[i] = splitFar;

		beMath::fmat4 splitView, splitProj, splitViewProj;
		beMath::fvec3 splitCenter;
		ComputeShadowMatrices(splitView, splitProj, splitViewProj, shadowSplit.PixelScale,
			splitCenter, shadowSplit.Near, shadowSplit.Far,
			camOrientation, perspectiveDesc.CamPos, perspectiveDesc.ViewProjMat,
			splitNear, splitFar,
			splitOrientation, ShadowRange, state.ShadowResolution,
			perspectiveDesc.Flags & PerspectiveFlags::Omnidirectional);

		shadowSplit.Proj = splitViewProj;

		PerspectiveDesc splitPerspectiveDesc(
				splitCenter,
				splitOrientation[0],
				splitOrientation[1],
				splitOrientation[2],
				splitView,
				splitProj,
				shadowSplit.Near,
				shadowSplit.Far,
				perspectiveDesc.Flipped,
				perspectiveDesc.Time,
				perspectiveDesc.TimeStep,
				i
			);
		shadowState.Perspective[i]->SetDesc(splitPerspectiveDesc);

		perspective.AddPerspective(shadowState.Perspective[i]);
	}
}

// Gets shadow maps.
beGraphics::TextureViewHandle GatherShadow(const LightInternals<DirectionalLightController>::ShadowState &shadowState)
{
	const beg::TextureTarget *pTarget = (shadowState.Perspective[0]) ? shadowState.Perspective[0]->GetPipe()->GetAnyTarget("SceneShadowTarget") : nullptr;
	return beg::Any::TextureViewHandle( (pTarget) ? pTarget->GetTexture() : nullptr );
}

template LightControllers<DirectionalLightController>;
template LightControllerBase<DirectionalLightController>;
template void InstantiateLightControllerFunctions<DirectionalLightController>();

} // namespace
