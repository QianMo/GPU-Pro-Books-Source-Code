#ifndef COMMON_XINAMED_H_INCLUDED
#define COMMON_XINAMED_H_INCLUDED

#include "Forw.h"
#include "Named.h"

namespace Mod
{
	class XINamed : public Named
	{
		// construction/ destruction
	public:
		XINamed( const XMLElemPtr& elem );
		~XINamed();

	};
}


#endif