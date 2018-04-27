#include "Precompiled.h"
#include "XINamed.h"
#include "XIAttribute.h"

namespace Mod
{
	XINamed::XINamed( const XMLElemPtr& elem ):
	Named( XIAttString( elem, L"name" ) )
	{
		
	}

	//------------------------------------------------------------------------

	XINamed::~XINamed()
	{

	}
}
