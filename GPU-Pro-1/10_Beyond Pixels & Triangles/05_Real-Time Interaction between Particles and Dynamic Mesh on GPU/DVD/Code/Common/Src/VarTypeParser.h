#ifndef COMMON_VARTYPEPARSER_H_INCLUDED
#define COMMON_VARTYPEPARSER_H_INCLUDED

#include "VarType.h"

namespace Mod
{
	class VarTypeParser
	{
		// types
	public:
		typedef Types2< String, VarType::Type > :: Map TypeMap;

		// construction/ destruction
	public:
		VarTypeParser();
		~VarTypeParser();

		// manipulation/ access
	public:
		VarType::Type GetItem( const String& val ) const;

		static VarTypeParser& Single(); 

		// data
	private:
		TypeMap mTypeMap;

	};
}


#endif