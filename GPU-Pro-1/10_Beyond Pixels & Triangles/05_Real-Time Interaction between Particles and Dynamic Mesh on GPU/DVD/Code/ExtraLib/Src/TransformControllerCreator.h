#ifndef EXTRALIB_TRANSFORMCONTROLLERCREATOR_H_INCLUDED
#define EXTRALIB_TRANSFORMCONTROLLERCREATOR_H_INCLUDED

#include "Forw.h"

#include "ItemCreator.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE TransformControllerCreatorNS
#include "ConfigurableImpl.h"

#define MD_NAMESPACE TransformControllerCreatorNS
#include "Singleton.h"

namespace Mod
{

	class TransformControllerCreator :		public ItemCreator
													< 
														DefaultItemCreatorConfigT
														<
															TransformControllerCreatorNS::ConfigurableImpl< TransformControllerCreatorConfig >, 
															TransformController,
															TransformControllerConfig,
															String
														>
													>,
											public TransformControllerCreatorNS::Singleton< TransformControllerCreator >
	{
		// types
	public:

		// constructors / destructors
	public:
		explicit TransformControllerCreator( const TransformControllerCreatorConfig& cfg );
		~TransformControllerCreator();
	
		// manipulation/ access
	public:
		TransformControllerPtr CreateItem( const XMLElemPtr& elem );
		using Parent::CreateItem;

		// data
	private:

	};

	//------------------------------------------------------------------------


}

#endif