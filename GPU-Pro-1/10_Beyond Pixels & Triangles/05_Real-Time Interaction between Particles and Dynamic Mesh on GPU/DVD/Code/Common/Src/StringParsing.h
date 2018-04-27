#ifndef COMMON_STRINGPARSING_H_INCLUDED
#define COMMON_STRINGPARSING_H_INCLUDED

namespace Mod
{
	String GetToken( const WCHAR*& str );
	String GetToken( const WCHAR* tokens[], UINT32 N, const WCHAR*& str );
	String GetToken( const WCHAR*& str, const WCHAR punctuation[], UINT32 N );

	template <typename T>
	T FromString( const String& val );
}

#endif