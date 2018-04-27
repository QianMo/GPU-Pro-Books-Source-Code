#include "Precompiled.h"

#include "Math/Src/Operations.h"

#include "Common/Src/XIAttribute.h"

#include "Common/Src/VarVariant.h"

#include "TransformControllerConfig.h"
#include "TransformController.h"

#define MD_NAMESPACE TransformControllerNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	using namespace Math;

	TransformController::TransformController( const TransformControllerConfig& cfg ) :
	Base( cfg ),
	mT( 0 ),
	mLoop( false )
	{
		mSpeed = XIAttFloat( cfg.elem, L"speed", 1.f );
		mLoop = XIAttInt( cfg.elem, L"loop", 0 ) ? true : false;
	}

	//------------------------------------------------------------------------

	TransformController::~TransformController() 
	{

	}

	//------------------------------------------------------------------------

	void
	TransformController::Update( float delta )
	{
		mT += delta * mSpeed;

		SetPosition( mT );
	}

	//------------------------------------------------------------------------

	void
	TransformController::SetPosition( float t )
	{
		if( mLoop )
		{
			mT = fmod( mT, 1.f );
		}
		else
		{
			mT = saturate( t );
		}

		SetPositionImpl( mT );
	}

	//------------------------------------------------------------------------

	float
	TransformController::GetPosition() const
	{
		return mT;
	}

	//------------------------------------------------------------------------

	void
	TransformController::UpdateNode( const NodePtr& node ) const
	{
		UpdateNodeImpl( node );
	}

	//------------------------------------------------------------------------

	void
	TransformController::Align( const NodePtr& node )
	{
		AlignImpl( mT, node );
	}

	//------------------------------------------------------------------------

	bool
	TransformController::IsFinished() const
	{
		return mT >= 1.f;
	}

	//------------------------------------------------------------------------

	void
	TransformController::AccountAspectRatio( float aspect )
	{
		AccountAspectRatioImpl( aspect );
	}

}