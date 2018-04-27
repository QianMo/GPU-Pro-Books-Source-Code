#ifndef EXTRALIB_EFFECTPARAMCONTROLLERCREATOR_H_INCLUDED
#define EXTRALIB_EFFECTPARAMCONTROLLERCREATOR_H_INCLUDED

#include "Forw.h"

#include "Common/Src/Forw.h"
#include "SceneRender/Src/Forw.h"

#include "ItemCreator.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE EffectParamControllerCreatorNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class EffectParamControllerCreator : public ItemCreator	<
																DefaultItemCreatorConfigT
																<
																	EffectParamControllerCreatorNS::ConfigurableImpl< EffectParamControllerCreatorConfig >, 
																	EffectParamController,
																	EffectParamControllerConfig,
																	String
																>
															>
	{
		// types
	public:

		// constructors / destructors
	public:
		explicit EffectParamControllerCreator( const EffectParamControllerCreatorConfig& cfg );
		~EffectParamControllerCreator();
	
		// manipulation/ access
	public:
		EffectParamControllerPtr CreateItem( const EffectParamControllerConfig& cfg );
		using Parent::CreateItem;

		// data
	private:

	};
}

#endif