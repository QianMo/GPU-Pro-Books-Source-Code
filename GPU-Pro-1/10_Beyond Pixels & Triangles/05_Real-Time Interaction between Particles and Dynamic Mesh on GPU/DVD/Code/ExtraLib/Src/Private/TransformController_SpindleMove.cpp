#include "Precompiled.h"

#include "Math/Src/Operations.h"

#include "Common/Src/XIElemAttribute.h"
#include "Common/Src/XIAttribute.h"

#include "Common/Src/VarVariant.h"
#include "Common/Src/TypedParam.h"

#include "SceneRender/Src/Node.h"

#include "TransformControllerConfig.h"
#include "TransformController_SpindleMove.h"

namespace Mod
{
	using namespace Math;

	//------------------------------------------------------------------------

	/*explicit*/
	TransformController_SpindleMove::TransformController_SpindleMove( const TransformControllerConfig& cfg ) :
	Parent( cfg ),
	mPosition( Math::ZERO_3 ),
	mOrient( Math::ZERO_4 )
	{
		mPosition_From	= XIVector3( cfg.elem, L"pos_from"	);
		mPosition_To	= XIVector3( cfg.elem, L"pos_to"	);
		mAxis			= XIVector3( cfg.elem, L"axis"		);

		mStartAgnle		= XIFloat( cfg.elem, L"start_angle", L"val"	);
		mEndAngle		= XIFloat( cfg.elem, L"end_angle", L"val"	);
	}

	//------------------------------------------------------------------------

	TransformController_SpindleMove::~TransformController_SpindleMove()
	{

	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	TransformController_SpindleMove::SetPositionImpl( float t ) /*OVERRIDE*/
	{
		float pos_t = saturate( t * 1.25f );

		const float ab = 0.85f;
		float ang_t = t < ab ? ( t / ab ) : ( 1.0f + 2.5f * ( t - ab ) * ( 1.f - t ) );

		mPosition	= lerp( mPosition_From, mPosition_To, pos_t );
		mOrient		= quatRotAxisAngle( mAxis, lerp( mStartAgnle, mEndAngle, ang_t ) * Math::PI_F / 180.f );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	TransformController_SpindleMove::UpdateNodeImpl( const NodePtr& node ) const /*OVERRIDE*/
	{
		node->SetPosition( mPosition );
		node->SetOrient( mOrient );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	TransformController_SpindleMove::AlignImpl( float t, const NodePtr& node )	/*OVERRIDE*/
	{
		float3 axis;
		float angle;
		
		quatDecompose( node->GetOrient(), axis, angle );

		angle *= 180.f / Math::PI_F * ( axis.y > 0 ? 1.f : -1.f );

		float angleDistance  = mEndAngle - mStartAgnle;

		mStartAgnle  = angle - t*angleDistance;
		mEndAngle = mStartAgnle + angleDistance;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	TransformController_SpindleMove::AccountAspectRatioImpl( float aspect )	/*OVERRIDE*/
	{
		mPosition_From.x	*= aspect;
		mPosition_To.x		*= aspect;
	}

}