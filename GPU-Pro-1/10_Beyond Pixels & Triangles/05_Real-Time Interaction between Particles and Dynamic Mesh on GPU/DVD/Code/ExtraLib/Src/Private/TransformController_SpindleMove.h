#ifndef EXTRALIB_TRANSFORMCONTROLLER_SPINDLEMOVE_H_INCLUDED
#define EXTRALIB_TRANSFORMCONTROLLER_SPINDLEMOVE_H_INCLUDED

#include "Math/Src/Types.h"

#include "Forw.h"

#include "TransformController.h"

namespace Mod
{

	class TransformController_SpindleMove : public TransformController
	{
		// types
	public:

		// constructors / destructors
	public:
		explicit TransformController_SpindleMove( const TransformControllerConfig& cfg );
		~TransformController_SpindleMove();
	
		// manipulation/ access
	public:

		// polymorphism
	private:
		virtual void SetPositionImpl( float t )							OVERRIDE;
		virtual void UpdateNodeImpl( const NodePtr& node ) const		OVERRIDE;
		virtual void AlignImpl( float t, const NodePtr& node )			OVERRIDE;
		virtual void AccountAspectRatioImpl( float aspect )				OVERRIDE;

		// data
	private:
		Math::float3	mPosition;
		Math::float4	mOrient;

		Math::float3	mPosition_From;
		Math::float3	mPosition_To;

		Math::float3	mAxis;
		
		float			mStartAgnle;
		float			mEndAngle;
	};
}

#endif