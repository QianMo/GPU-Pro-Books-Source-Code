#ifndef COMMON_STRINGUTILS_H_INCLUDED
#define COMMON_STRINGUTILS_H_INCLUDED

namespace Mod
{
	String AsString( const String& string );
	String AsString( int number );
	String AsString( UINT32 number );
	String AsString( float number, UINT32 digitsAfterSep = 2 );

	String AsLower( const String& string );

	void SplitString( const String& string, WCHAR separator, Strings& oStrings );

	int AsInt( const String& string );
}

#endif