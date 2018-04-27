#ifndef EXTRALIB_CAMERACONTROLLER_FOLLOWTARGET_H_INCLUDED
#define EXTRALIB_CAMERACONTROLLER_FOLLOWTARGET_H_INCLUDED

#include "Math/Src/Forw.h"

#include "Forw.h"

#include "CameraController.h"

namespace Mod
{
	class CameraController_FollowTarget : public CameraController
	{
		// types
	public:
		typedef TargetedCameraControllerConfig ConfigType;

	public:
		explicit CameraController_FollowTarget( const TargetedCameraControllerConfig& cfg );
		~CameraController_FollowTarget();

		// manipulation / access
	public:
		void SetDistance( float distance );

		void SetRotation( float rx, float ry );

		void SetDisplacement( float dx, float dy, float dz );

		void SetTarget( const ENodePtr& target );

		// polymorphism
	private:
		virtual void UpdateImpl( float dt ) OVERRIDE;

		// data
	private:
		float	mDistance;
		float	mTargetDistance;

		float	mX_DSP;
		float	mY_DSP;
		float	mZ_DSP;

		float	mRotX;
		float	mRotY;

		Math::float3 mDSP;
	};
}

#endif