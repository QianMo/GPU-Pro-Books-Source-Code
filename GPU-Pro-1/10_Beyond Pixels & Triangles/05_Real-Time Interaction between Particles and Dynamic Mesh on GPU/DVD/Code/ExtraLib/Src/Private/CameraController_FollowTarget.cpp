#include "Precompiled.h"

#include "Math/Src/BBox.h"
#include "Math/Src/Operations.h"

#include "SceneRender/Src/CameraConfig.h"
#include "SceneRender/Src/Camera.h"
#include "SceneRender/Src/ENode.h"

#include "CameraControllerConfig.h"

#include "CameraController_FollowTarget.h"

namespace Mod
{
	using namespace Math;

	/*explicit*/
	CameraController_FollowTarget::CameraController_FollowTarget( const TargetedCameraControllerConfig& cfg ) :
	Parent ( cfg ),
	mDistance( 0 ),
	mTargetDistance( 0 ),
	mX_DSP( 0 ),
	mY_DSP( 0 ),
	mZ_DSP( 0 ),
	mRotX( 0 ),
	mRotY( 0 ),
	mDSP( 0, 0, 0 )
	{
		SetTarget( cfg.target );
	}

	//------------------------------------------------------------------------


	CameraController_FollowTarget::~CameraController_FollowTarget()
	{

	}

	//------------------------------------------------------------------------

	void
	CameraController_FollowTarget::SetDistance( float distance )
	{
		mTargetDistance = distance;
	}

	//------------------------------------------------------------------------

	void
	CameraController_FollowTarget::SetRotation( float rx, float ry )
	{
		mRotX = rx;
		mRotY = ry;
	}

	//------------------------------------------------------------------------

	void
	CameraController_FollowTarget::SetDisplacement( float dx, float dy, float dz )
	{
		mX_DSP = dx;
		mY_DSP = dy;
		mZ_DSP = dz;
	}

	//------------------------------------------------------------------------

	void
	CameraController_FollowTarget::SetTarget( const ENodePtr& target )
	{
		static_cast<TargetedCameraControllerConfig&>( config() ).target = target;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	CameraController_FollowTarget::UpdateImpl( float dt ) /*OVERRIDE*/
	{
		const ConfigType& cfg = static_cast<const ConfigType&>( GetConfig() );
		const CameraPtr& cam = cfg.camera;

		float3 target( 0, 0, 0 );

		if( cfg.target )
		{
			target = mul( float4( target, 1 ), cfg.target->GetGlobalTransform() );
		}

		const Camera::ConfigType& ccfg = cam->GetConfig();

		float3 cpos = ccfg.position;

		float3 extraDisplace( mX_DSP, mY_DSP, mZ_DSP );

		float3x4 rotMat = m3x4RotYawPitchRoll( mRotX, mRotY, 0 );

		if( cfg.target )
		{
			rotMat = mul( rotMat, m3x4RotQuat( cfg.target->GetOrient() ) );
		}

		float distance	= length( cpos - target - mDSP );

		mDistance		= lerp( distance, mTargetDistance, saturate( dt * 8 ) );

		float3 displace = mul( float3(0,0,mDistance), rotMat );

		mDSP = mul( extraDisplace, rotMat );

		cam->SetPosition( target + mDSP - displace );
		cam->SetOrient( quatRotMat( rotMat ) );
	}

	//------------------------------------------------------------------------

}