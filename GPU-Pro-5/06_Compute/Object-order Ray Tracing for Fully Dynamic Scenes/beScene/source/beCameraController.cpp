/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beCameraController.h"
#include <beEntitySystem/beEntities.h>
#include <beEntitySystem/beAnimatedHost.h>

#include "beScene/bePipelinePerspective.h"

#include <beCore/beReflectionProperties.h>

#include <beMath/beVector.h>
#include <beMath/beMatrix.h>

namespace beScene
{

BE_CORE_PUBLISH_COMPONENT(CameraController)

const beCore::ReflectionProperty CameraControllerProperties[] =
{
	beCore::MakeReflectionProperty<float>("fov", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&CameraController::SetFOV) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&CameraController::GetFOV) ),
	beCore::MakeReflectionProperty<float>("aspect", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&CameraController::SetAspect) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&CameraController::GetAspect) ),
	beCore::MakeReflectionProperty<float>("near", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&CameraController::SetNearPlane) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&CameraController::GetNearPlane) ),
	beCore::MakeReflectionProperty<float>("far", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&CameraController::SetFarPlane) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&CameraController::GetFarPlane) ),
	beCore::MakeReflectionProperty<float>("time", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&CameraController::SetTime) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&CameraController::GetTime) ),
	beCore::MakeReflectionProperty<float>("time step", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&CameraController::SetTimeStep) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&CameraController::GetTimeStep) )
};
BE_CORE_ASSOCIATE_PROPERTIES(CameraController, CameraControllerProperties)

// Data defaults.
CameraController::Parameters::Parameters()
	: transform(beMath::fmat4::identity),
	view(beMath::fmat4::identity),
	proj(beMath::fmat4::identity),
	fov(1.0f),
	aspect(1.0f),
	nearPlane(0.2f),
	farPlane(16000.0f),

	time(0.0f),
	timeStep(0.0f)
{
}

// Constructor.
CameraController::CameraController(bees::AnimatedHost *pAnimationHost)
	: m_pAnimationHost(pAnimationHost),
	m_pPerspective(nullptr)
{
}

// Copy constructor.
CameraController::CameraController(const CameraController &right)
	// MONITOR: REDUNDANT
	: m_params(right.m_params),
	m_pAnimationHost(right.m_pAnimationHost)
	// MONITOR: Deliberately not cloning perspective?
{
}

// Destructor.
CameraController::~CameraController()
{
}

// Synchronizes this controller with the controlled entity.
void CameraController::Flush(const bees::EntityHandle entity)
{
	using beEntitySystem::Entities;

	const beMath::fvec3 &pos = Entities::GetPosition(entity);
	const beMath::fmat3 &orientation = Entities::GetOrientation(entity);

	m_params.transform = beMath::mat_transform(pos, orientation[2], orientation[1], orientation[0]);
	m_params.view = beMath::mat_view(pos, orientation[2], orientation[1], orientation[0]);
	m_params.proj = beMath::mat_proj(m_params.fov, m_params.aspect, m_params.nearPlane, m_params.farPlane);

	if (m_pPerspective)
		m_pPerspective->SetDesc(PerspectiveFromCamera(this));
}

/// Sets the perspective.
void CameraController::SetPerspective(PipelinePerspective *pPerspective)
{
	m_pPerspective = pPerspective;
}

// Steps this controller.
void CameraController::Step(float timeStep)
{
	m_params.timeStep = timeStep;
	m_params.time += m_params.timeStep;
}

// Attaches this controller to the scenery.
void CameraController::Attach(beEntitySystem::Entity *entity)
{
	if (m_pAnimationHost)
		m_pAnimationHost->AddAnimated(this);
}

// Detaches this controller from the scenery.
void CameraController::Detach(beEntitySystem::Entity *entity)
{
	if (m_pAnimationHost)
		m_pAnimationHost->RemoveAnimated(this);
}

// Clones this entity controller.
CameraController* CameraController::Clone() const
{
	return new CameraController(*this);
}

// Constructs a perspective description from the given camera controller.
PerspectiveDesc PerspectiveFromCamera(const CameraController *pCamera)
{
	return PerspectiveDesc(
		beMath::fvec3(pCamera->GetMatrix()[3]),
		beMath::fvec3(pCamera->GetMatrix()[0]),
		beMath::fvec3(pCamera->GetMatrix()[1]),
		beMath::fvec3(pCamera->GetMatrix()[2]),
		pCamera->GetViewMatrix(),
		pCamera->GetProjMatrix(),
		pCamera->GetNearPlane(),
		pCamera->GetFarPlane(),
		false,
		pCamera->GetTime(),
		pCamera->GetTimeStep());
}

} // namespace
