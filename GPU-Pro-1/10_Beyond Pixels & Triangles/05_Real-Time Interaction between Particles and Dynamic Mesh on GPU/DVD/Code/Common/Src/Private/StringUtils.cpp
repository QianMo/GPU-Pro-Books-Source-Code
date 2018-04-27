#include "Precompiled.h"

#include "StringUtils.h"

namespace Mod
{
	String AsString( const String& string )
	{
		return string;
	}

	//------------------------------------------------------------------------

	String AsString( int number )
	{
		std::wostringstream oss;
		oss << number;

		return oss.str();
	}

	//------------------------------------------------------------------------

	String AsString( UINT32 number )
	{
		std::wostringstream oss;
		oss << number;

		return oss.str();
	}

	//------------------------------------------------------------------------

	String AsString( float number, UINT32 digitsAfterSep /*= 2*/ )
	{
		WCHAR str[64];
		WCHAR digs = static_cast<char>( std::min( digitsAfterSep, (UINT32)9 ) );
		WCHAR fmt[] = L"%._f";

		fmt[2] = static_cast<WCHAR>( '0' + digs );

		_snwprintf( str, sizeof(str) / sizeof(str[0]), fmt, number );

		return str;
	}

	//------------------------------------------------------------------------

	String AsLower( const String& string )
	{
		String result( string );
		ToLower( result );

		return result;
	}

	//------------------------------------------------------------------------

	void SplitString( const String& string, WCHAR separator, Strings& oStrings )
	{
		Types< WCHAR > :: Vec charArray( string.size() + 1 );
		wcscpy( &charArray[0], string.c_str() );

		WCHAR separators[] = { separator, '\0' };

		WCHAR* token = wcstok( &charArray[0], separators );

		oStrings.push_back( token );

		while( WCHAR* token = wcstok( NULL, separators ) )
		{
			oStrings.push_back( token );
		}
	}

	//------------------------------------------------------------------------

	int AsInt( const String& string )
	{
		return _wtoi( string.c_str() );
	}
}