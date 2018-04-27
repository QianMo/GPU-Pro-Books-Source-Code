#ifndef EXTRALIB_TRANSFORMCONTROLLER_SPINDLEWOBBLE_H_INCLUDED
#define EXTRALIB_TRANSFORMCONTROLLER_SPINDLEWOBBLE_H_INCLUDED

#include "Forw.h"

#include "Math/Src/Types.h"

#include "TransformController.h"

namespace Mod
{

	class TransformController_SpindleWobble : public TransformController
	{
		// types
	public:

		// constructors / destructors
	public:
		explicit TransformController_SpindleWobble( const TransformControllerConfig& cfg );
		~TransformController_SpindleWobble();
	
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
		Math::float3	mPositionCentre;
		Math::float3	mAxis;

		Math::float4	mOrient;
		Math::float3	mPos;

		float			mVertRange;		
	};
}

#endif