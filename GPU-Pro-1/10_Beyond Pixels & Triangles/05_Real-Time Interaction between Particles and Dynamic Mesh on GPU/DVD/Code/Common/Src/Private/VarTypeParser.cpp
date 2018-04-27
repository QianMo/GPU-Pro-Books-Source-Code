#include "Precompiled.h"
#include "VarTypeParser.h"

namespace Mod
{

	VarTypeParser::VarTypeParser()
	{
		using namespace VarType;

#define MD_ADD_TYPE(name) { String val = L#name; ToLower( val ); mTypeMap.insert( TypeMap::value_type( val, name ) ); }
#define MD_ADD_TYPE_LINE(name)	MD_ADD_TYPE(name)		MD_ADD_TYPE(name##2)		MD_ADD_TYPE( name##3 )		MD_ADD_TYPE( name##4 )		MD_ADD_TYPE( name##2x2 )		MD_ADD_TYPE( name##3x3 )		MD_ADD_TYPE( name##4x4 )		MD_ADD_TYPE( name##3x4 )\
								MD_ADD_TYPE(name##_VEC) MD_ADD_TYPE(name##2_VEC)	MD_ADD_TYPE( name##3_VEC )	MD_ADD_TYPE( name##4_VEC )	MD_ADD_TYPE( name##2x2_VEC )	MD_ADD_TYPE( name##3x3_VEC )	MD_ADD_TYPE( name##4x4_VEC )	MD_ADD_TYPE( name##3x4_VEC )
		MD_ADD_TYPE_LINE(FLOAT) 
		MD_ADD_TYPE_LINE(INT)
		MD_ADD_TYPE_LINE(UINT)

#undef MD_ADD_TYPE_LINE
#undef MD_ADD_TYPE

		mTypeMap.insert( TypeMap::value_type( L"SR", VarType::SHADER_RESOURCE ) );

	}

	//------------------------------------------------------------------------

	VarTypeParser::~VarTypeParser()
	{

	}

	//------------------------------------------------------------------------

	VarType::Type
	VarTypeParser::GetItem( const String& val ) const
	{
		TypeMap::const_iterator found = mTypeMap.find( val );
		MD_FERROR_ON_FALSE( found != mTypeMap.end() );

		return found->second;
	}

	//------------------------------------------------------------------------

	/*static*/
	VarTypeParser&
	VarTypeParser::Single()
	{
		static VarTypeParser parser;
		return parser;
	}


}