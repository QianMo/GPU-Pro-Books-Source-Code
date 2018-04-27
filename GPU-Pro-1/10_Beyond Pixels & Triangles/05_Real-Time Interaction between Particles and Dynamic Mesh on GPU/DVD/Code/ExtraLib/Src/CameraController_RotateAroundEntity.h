#ifndef EXTRALIB_CAMERACONTROLLER_ROTATEAROUNDENTITY_H_INCLUDED
#define EXTRALIB_CAMERACONTROLLER_ROTATEAROUNDENTITY_H_INCLUDED

#include "Math/Src/Forw.h"

#include "Forw.h"

#include "CameraController.h"

namespace Mod
{
	class CameraController_RotateAroundEntity : public CameraController
	{
		// types
	public:
		typedef TargetedCameraControllerConfig ConfigType;

	public:
		explicit CameraController_RotateAroundEntity( const TargetedCameraControllerConfig& cfg );
		~CameraController_RotateAroundEntity();

		// manipulation / access
	public:
		void AdjustDistance( float amount );
		void RotateX( float amount );
		void RotateY( float amount );

		void ResetRotation();

		void SetDisplacement( float dx, float dy, float dz );

		void SetRotationEnabled( bool enabled );

		void SetTarget( const ENodePtr& target, bool reset_distance );

		// polymorphism
	private:
		virtual void UpdateImpl( float dt ) OVERRIDE;

		// data
	private:
		float	mDistance;
		float	mTargetDistance;
		float	mRotX;
		float	mRotY;

		float	mX_DSP;
		float	mY_DSP;
		float	mZ_DSP;

		bool	mRotationEnabled;

		Math::float3 mDSP;
	};
}

#endif