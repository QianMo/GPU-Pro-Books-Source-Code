/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_CAMERA_CONTROLLER
#define BE_SCENE_CAMERA_CONTROLLER

#include "beScene.h"
#include <beEntitySystem/beEntityController.h>
#include <beEntitySystem/beSynchronized.h>
#include <beEntitySystem/beAnimated.h>
#include <beMath/beMatrixDef.h>
#include <lean/smart/resource_ptr.h>
#include <beScene/bePerspective.h>

namespace beEntitySystem
{
	class AnimatedHost;
}

namespace beScene
{

// Prototypes
class Pipe;
class PipelinePerspective;

/// Camera controller.
class CameraController : public beEntitySystem::SingularEntityController, public beEntitySystem::Animated
{
public:
	struct Parameters
	{
		beMath::fmat4 transform;
		beMath::fmat4 view;
		beMath::fmat4 proj;

		float fov;
		float aspect;
		float nearPlane;
		float farPlane;

		float time;
		float timeStep;

		/// Data defaults.
		BE_SCENE_API Parameters();
	};

private:
	Parameters m_params;

	beEntitySystem::AnimatedHost *m_pAnimationHost;
	lean::com_ptr<PipelinePerspective> m_pPerspective;

public:
	/// Constructor.
	BE_SCENE_API CameraController(beEntitySystem::AnimatedHost *pAnimationHost);
	/// Copy constructor.
	BE_SCENE_API CameraController(const CameraController &right);
	/// Destructor.
	BE_SCENE_API ~CameraController();

	/// Sets the field of view.
	LEAN_INLINE void SetFOV(float fov) { m_params.fov = fov; }
	/// Gets the field of view.
	LEAN_INLINE float GetFOV() const { return m_params.fov; }

	/// Sets the aspect ratio.
	LEAN_INLINE void SetAspect(float aspect) { m_params.aspect = aspect; }
	/// Gets the aspect ratio.
	LEAN_INLINE float GetAspect() const { return m_params.aspect; }

	/// Sets the near plane.
	LEAN_INLINE void SetNearPlane(float nearPlane) { m_params.nearPlane = nearPlane; }
	/// Gets the near plane.
	LEAN_INLINE float GetNearPlane() const { return m_params.nearPlane; }

	/// Sets the far plane.
	LEAN_INLINE void SetFarPlane(float farPlane) { m_params.farPlane = farPlane; }
	/// Gets the far plane.
	LEAN_INLINE float GetFarPlane() const { return m_params.farPlane; }

	/// Sets the time.
	LEAN_INLINE void SetTime(float time) { m_params.time = time; }
	/// Gets the time.
	LEAN_INLINE float GetTime() const { return m_params.time; }

	/// Sets the time step.
	LEAN_INLINE void SetTimeStep(float timeStep) { m_params.timeStep = timeStep; }
	/// Gets the time step.
	LEAN_INLINE float GetTimeStep() const { return m_params.timeStep; }

	/// Synchronizes this controller with the controlled entity.
	BE_SCENE_API void Flush(const beEntitySystem::EntityHandle entity);
	/// Steps this controller.
	BE_SCENE_API void Step(float timeStep);

	/// Attaches this controller.
	BE_SCENE_API void Attach(beEntitySystem::Entity *entity);
	/// Detaches this controller.
	BE_SCENE_API void Detach(beEntitySystem::Entity *entity);

	/// Gets the transformation matrix.
	LEAN_INLINE const beMath::fmat4& GetMatrix() const { return m_params.transform; } 
	/// Gets the view matrix.
	LEAN_INLINE const beMath::fmat4& GetViewMatrix() const { return m_params.view; }
	/// Gets the projection matrix.
	LEAN_INLINE const beMath::fmat4& GetProjMatrix() const { return m_params.proj; }

	/// Sets the perspective.
	BE_SCENE_API void SetPerspective(PipelinePerspective *pPerspective);
	/// Gets the perspective.
	LEAN_INLINE PipelinePerspective* GetPerspective() const { return m_pPerspective; }

	/// Gets the reflection properties.
	BE_SCENE_API static Properties GetOwnProperties();
	/// Gets the reflection properties.
	BE_SCENE_API Properties GetReflectionProperties() const;

	/// Clones this entity controller.
	BE_SCENE_API CameraController* Clone() const;

	/// Gets the controller type.
	BE_SCENE_API static const beCore::ComponentType* GetComponentType();
	/// Gets the controller type.
	BE_SCENE_API const beCore::ComponentType* GetType() const;
};

/// Constructs a perspective description from the given camera controller.
/// @relates CameraController
BE_SCENE_API PerspectiveDesc PerspectiveFromCamera(const CameraController *pCamera);

} // namespace

#endif