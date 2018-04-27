#include "Precompiled.h"

#include "Common/Src/XIElemAttribute.h"
#include "Common/Src/XIAttribute.h"

#include "Math/Src/Operations.h"

#include "SceneRender/Src/Node.h"

#include "TransformControllerConfig.h"
#include "TransformController_SpindleWobble.h"

namespace Mod
{
	using namespace Math;

	TransformController_SpindleWobble::TransformController_SpindleWobble( const TransformControllerConfig& cfg ) :
	Parent( cfg ),
	mOrient( IDENTITY_QUAT ),
	mPos ( ZERO_4 )
	{
		mPositionCentre	= XIVector3( cfg.elem, L"centre"	);
		mAxis			= XIVector3( cfg.elem, L"axis"		);

		mVertRange		= XIFloat( cfg.elem, L"vert_range", L"val"	);
	}

	//------------------------------------------------------------------------

	TransformController_SpindleWobble::~TransformController_SpindleWobble() 
	{
	}

	//------------------------------------------------------------------------

	/*virtual*/
	void
	TransformController_SpindleWobble::SetPositionImpl( float t ) /*OVERRIDE*/
	{
		mOrient	= quatRotAxisAngle( mAxis, t * TWO_PI_F );
		mPos	= mPositionCentre + mAxis * mVertRange * sinf( t * TWO_PI_F );
	}

	//------------------------------------------------------------------------

	/*virtual*/
	void
	TransformController_SpindleWobble::UpdateNodeImpl( const NodePtr& node ) const /*OVERRIDE*/
	{
		node->SetOrient( mOrient );
		node->SetPosition( mPos );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	TransformController_SpindleWobble::AlignImpl( float t, const NodePtr& node ) /*OVERRIDE*/
	{
		t;
		mPositionCentre = node->GetPosition();
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	TransformController_SpindleWobble::AccountAspectRatioImpl( float aspect ) /*OVERRIDE*/
	{
		mPositionCentre.x *= aspect;
	}

}