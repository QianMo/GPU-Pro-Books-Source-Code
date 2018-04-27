#include "Precompiled.h"

#include "Math/Src/BBox.h"
#include "Math/Src/Operations.h"

#include "SceneRender/Src/CameraConfig.h"
#include "SceneRender/Src/Camera.h"
#include "SceneRender/Src/ENode.h"

#include "CameraControllerConfig.h"

#include "CameraController_RotateAroundEntity.h"

namespace Mod
{
	using namespace Math;

	/*explicit*/
	CameraController_RotateAroundEntity::CameraController_RotateAroundEntity( const TargetedCameraControllerConfig& cfg ) :
	Parent ( cfg ),
	mDistance( 0 ),
	mTargetDistance( 0 ),
	mRotX( 0 ),
	mRotY( 0 ),
	mX_DSP( 0 ),
	mY_DSP( 0 ),
	mZ_DSP( 0 ),
	mDSP( 0, 0, 0 ),
	mRotationEnabled( true )
	{
		SetTarget( cfg.target, true );
	}

	//------------------------------------------------------------------------

	
	CameraController_RotateAroundEntity::~CameraController_RotateAroundEntity()
	{

	}

	//------------------------------------------------------------------------

	void
	CameraController_RotateAroundEntity::AdjustDistance( float amount )
	{
		mTargetDistance += amount;
	}

	//------------------------------------------------------------------------

	void
	CameraController_RotateAroundEntity::RotateX( float amount )
	{
		mRotX += amount;
	}

	//------------------------------------------------------------------------

	void
	CameraController_RotateAroundEntity::RotateY( float amount )
	{
		mRotY += amount;
	}

	//------------------------------------------------------------------------

	void
	CameraController_RotateAroundEntity::ResetRotation()
	{
		mRotX = 0;
		mRotY = 0;
	}

	//------------------------------------------------------------------------

	void
	CameraController_RotateAroundEntity::SetDisplacement( float dx, float dy, float dz )
	{
		mX_DSP = dx;
		mY_DSP = dy;
		mZ_DSP = dz;
	}

	//------------------------------------------------------------------------

	void
	CameraController_RotateAroundEntity::SetRotationEnabled( bool enabled )
	{
		mRotationEnabled = enabled;
	}

	//------------------------------------------------------------------------

	namespace
	{
		BBox getScaledBBox( const ENodePtr& node )
		{
			const BBox& bbox = node->GetBBox();
			const float3& scale = node->GetScale();
			return BBox( bbox.GetMin() * scale, bbox.GetMax() * scale );
		}
	}

	void
	CameraController_RotateAroundEntity::SetTarget( const ENodePtr& target, bool reset_distance )
	{
		static_cast<TargetedCameraControllerConfig&>( config() ).target = target;

		if( target && reset_distance )
		{
			mTargetDistance = get_radius( getScaledBBox( target ) ) * 8;
			mDistance = mTargetDistance;
		}
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	CameraController_RotateAroundEntity::UpdateImpl( float dt ) /*OVERRIDE*/
	{
		const ConfigType& cfg = static_cast<const ConfigType&>( GetConfig() );
		const CameraPtr& cam = cfg.camera;

		BBox bbox = cfg.target ? cfg.target->GetBBox() : Math::BBox( float3(-1, -1, -1), float3(+1, +1, +1) );

		if( cfg.target )
		{
			bbox = getScaledBBox( cfg.target );
		}

		// clamp target distance
		mTargetDistance = std::max( get_radius( bbox) * 4, mTargetDistance ) ;

		float3 target = get_center( bbox );

		if( cfg.target )
		{
			target = mul( float4( target, 1 ), cfg.target->GetGlobalTransform() );
		}

		const Camera::ConfigType& ccfg = cam->GetConfig();

		float3 cpos = ccfg.position;

		float3 extraDisplace( mX_DSP, mY_DSP, mZ_DSP );

		if( mRotationEnabled )
		{
			mRotX += dt;
		}

		float3x4 rotMat = m3x4RotYawPitchRoll( mRotX, mRotY, 0 );

		float distance	= length( cpos - target - mDSP );

		mDistance		= lerp( distance, mTargetDistance, saturate( dt * 8 ) );

		float3 displace = mul( float3(0,0,mDistance), rotMat );

		mDSP = mul( extraDisplace, rotMat );

		cam->SetPosition( target + mDSP - displace );
		cam->SetOrient( quatRotMat( rotMat ) );
	}



	//------------------------------------------------------------------------

}