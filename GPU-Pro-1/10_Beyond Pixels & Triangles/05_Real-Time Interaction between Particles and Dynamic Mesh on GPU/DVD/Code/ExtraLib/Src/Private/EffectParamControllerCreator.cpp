#include "Precompiled.h"

#include "Common/Src/XIAttribute.h"

#include "SceneRender/Src/EffectParamControllerConfig.h"

#include "EffectParamControllerCreatorConfig.h"
#include "EffectParamControllerCreator.h"
#include "EffectParamControllerCreateFunctions.h"

#include "ItemCreator.cpp.h"

#define MD_NAMESPACE EffectParamControllerCreatorNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	EffectParamControllerCreator::EffectParamControllerCreator( const EffectParamControllerCreatorConfig& cfg ) :
	Parent( cfg )
	{
		AddCreator( L"animated_texture", CreateEffectParamController_AnimatedTexture );
		AddCreator( L"linear_float", CreateEffectParamController_LinearFloat );
		AddCreator( L"linear_float2", CreateEffectParamController_LinearFloat2 );
		AddCreator( L"linear_float3", CreateEffectParamController_LinearFloat3 );
		AddCreator( L"linear_float4", CreateEffectParamController_LinearFloat4 );

		AddCreator( L"user_float",	CreateEffectParamController_UserFloat	);
		AddCreator( L"user_float2",	CreateEffectParamController_UserFloat2	);
		AddCreator( L"user_float3",	CreateEffectParamController_UserFloat3	);
		AddCreator( L"user_float4",	CreateEffectParamController_UserFloat4	);
	}

	//------------------------------------------------------------------------

	EffectParamControllerCreator::~EffectParamControllerCreator() 
	{
	}

	//------------------------------------------------------------------------

	EffectParamControllerPtr
	EffectParamControllerCreator::CreateItem( const EffectParamControllerConfig& cfg )
	{
		XIAttString type( cfg.elem, L"type" );

		return CreateItem( type, cfg );
	}

	//------------------------------------------------------------------------

}