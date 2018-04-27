#include "Precompiled.h"

#include "Common/Src/XIAttribute.h"

#include "TransformControllerConfig.h"
#include "TransformController.h"

#include "TransformControllerCreatorConfig.h"
#include "TransformControllerCreator.h"

#include "TransformControllerCreateFunctions.h"

#define MD_NAMESPACE TransformControllerCreatorNS
#include "ConfigurableImpl.cpp.h"

#define MD_NAMESPACE TransformControllerCreatorNS
#include "Singleton.cpp.h"

#include "ItemCreator.cpp.h"

#include "BlankExpImp.h"

namespace Mod
{
	template class TransformControllerCreatorNS::ConfigurableImpl< TransformControllerCreatorConfig >;
	template class TransformControllerCreatorNS::Singleton< TransformControllerCreator >;
	template class ItemCreator< TransformControllerCreator::TConfig >;

	TransformControllerCreator::TransformControllerCreator( const TransformControllerCreatorConfig& cfg ) :
	Parent( cfg )
	{
		AddCreator( L"SpindleMove", CreateTransformController_SpindleMove );
		AddCreator( L"SpindleWobble", CreateTransformController_SpindleWobble );
	}

	//------------------------------------------------------------------------

	TransformControllerCreator::~TransformControllerCreator() 
	{
	}

	//------------------------------------------------------------------------

	TransformControllerPtr
	TransformControllerCreator::CreateItem( const XMLElemPtr& elem )
	{
		TransformControllerConfig cfg;

		cfg.elem = elem;

		return CreateItem( XIAttString( elem, L"type" ), cfg );
	}
}