#include "Precompiled.h"
#include "StringParsing.h"

#include "Math/Src/Types.h"
#include "Math/Src/Operations.h"


namespace Mod
{
	using namespace Math;

	String GetToken( const WCHAR* tokens[], UINT32 N, const WCHAR*& str )
	{
		size_t totalLen = wcslen( str );

		for( UINT32 i = 0; i < N; i ++ )
		{
			size_t tokenLen = wcslen( tokens[i] );
			if( totalLen < tokenLen )
				continue;

			if( !memcmp( tokens[i], str, tokenLen * sizeof(WCHAR) ) )
			{
				str += tokenLen;
				return tokens[i];
			}
		}

		return String();
	}

	//------------------------------------------------------------------------

	String GetToken( const WCHAR*& str, const WCHAR punctuation[], UINT32 N )
	{
		const WCHAR* a_str = str;
		while( WCHAR ch = *a_str )
		{
			for( UINT32 i = 0; i < N; i ++ )
				if( punctuation[i] == ch )
				{
					std::swap( str, a_str );
					return String( a_str, str );
				}

			a_str++;
		}

		std::swap( str, a_str );
		return String( a_str, str );
	}

	//------------------------------------------------------------------------

	template <>
	float FromString( const String& val )
	{
		return (float)_wtof( val.c_str() );
	}

	template <>
	INT32 FromString( const String& val )
	{
		return _wtoi( val.c_str() );
	}

	template <>
	UINT32 FromString( const String& val )
	{
		return (UINT32)_wtoi( val.c_str() );
	}

	namespace
	{

		template < typename T, const WCHAR* fmt >
		T FromString_4x4( const String& val )
		{
			T res( 0,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0 );
			swscanf(	val.c_str(), fmt, 
						&res[0][0], &res[1][0], &res[2][0], &res[3][0],
						&res[0][1], &res[1][1], &res[2][1], &res[3][1],
						&res[0][2], &res[1][2], &res[2][2], &res[3][2],
						&res[0][3], &res[1][3], &res[2][3], &res[3][3]
			);
			return res;
		}

		template < typename T, const WCHAR* fmt >
		T FromString_3x3( const String& val )
		{
			T res( 0,0,0,  0,0,0,  0,0,0 );
			swscanf(	val.c_str(), fmt, 
						&res[0][0], &res[1][0], &res[2][0],
						&res[0][1], &res[1][1], &res[2][1],
						&res[0][2], &res[1][2], &res[2][2]
			);
			return res;
		}

		template < typename T, const WCHAR* fmt >
		T FromString_2x2( const String& val )
		{
			T res( 0,0, 0,0 );
			swscanf(	val.c_str(), fmt, 
						&res[0][0], &res[1][0], 
						&res[0][1], &res[1][1]
			);
			return res;
		}

		template < typename T, const WCHAR* fmt >
		T FromString_4( const String& val )
		{
			T res( 0, 0, 0, 0);
			swscanf( val.c_str(), fmt, &res[0], &res[1], &res[2], &res[3] );
			return res;
		}

		template < typename T, const WCHAR* fmt >
		T FromString_3( const String& val )
		{
			T res( 0, 0, 0 );
			swscanf( val.c_str(), fmt, &res[0], &res[1], &res[2] );
			return res;
		}

		template < typename T, const WCHAR* fmt >
		T FromString_2( const String& val )
		{
			T res( 0, 0 );
			swscanf( val.c_str(), fmt, &res[0], &res[1] );
			return res;
		}
	}

	namespace FMT
	{
		extern const WCHAR float4x4_str[] = L"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f";
		extern const WCHAR float3x3_str[] = L"%f,%f,%f,%f,%f,%f,%f,%f,%f";
		extern const WCHAR float2x2_str[] = L"%f,%f,%f,%f";

		extern const WCHAR int4x4_str[] = L"%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d";
		extern const WCHAR int3x3_str[] = L"%d,%d,%d,%d,%d,%d,%d,%d,%d";
		extern const WCHAR int2x2_str[] = L"%d,%d,%d,%d";

		extern const WCHAR uint4x4_str[] = L"%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u";
		extern const WCHAR uint3x3_str[] = L"%u,%u,%u,%u,%u,%u,%u,%u,%u";
		extern const WCHAR uint2x2_str[] = L"%u,%u,%u,%u";

		extern const WCHAR float4_str[] = L"%f,%f,%f,%f";
		extern const WCHAR int4_str[] = L"%d,%d,%d,%d";
		extern const WCHAR uint4_str[] = L"%u,%u,%u,%u";

		extern const WCHAR float3_str[] = L"%f,%f,%f";
		extern const WCHAR int3_str[] = L"%d,%d,%d";
		extern const WCHAR uint3_str[] = L"%u,%u,%u";

		extern const WCHAR float2_str[] = L"%f,%f";
		extern const WCHAR int2_str[] = L"%d,%d";
		extern const WCHAR uint2_str[] = L"%u,%u";
	}

#define MD_DEFINE_FS_SPEC(func,type)				\
	template <>										\
	type FromString( const String& val )			\
	{												\
	return func< type, FMT::type##_str> ( val );	\
	}

	MD_DEFINE_FS_SPEC(FromString_4x4,float4x4)
	MD_DEFINE_FS_SPEC(FromString_4x4,int4x4)
	MD_DEFINE_FS_SPEC(FromString_4x4,uint4x4)

	MD_DEFINE_FS_SPEC(FromString_3x3,float3x3)
	MD_DEFINE_FS_SPEC(FromString_3x3,int3x3)
	MD_DEFINE_FS_SPEC(FromString_3x3,uint3x3)

	MD_DEFINE_FS_SPEC(FromString_2x2,float2x2)
	MD_DEFINE_FS_SPEC(FromString_2x2,int2x2)
	MD_DEFINE_FS_SPEC(FromString_2x2,uint2x2)

	MD_DEFINE_FS_SPEC(FromString_4,float4)
	MD_DEFINE_FS_SPEC(FromString_4,int4)
	MD_DEFINE_FS_SPEC(FromString_4,uint4)

	MD_DEFINE_FS_SPEC(FromString_3,float3)
	MD_DEFINE_FS_SPEC(FromString_3,int3)
	MD_DEFINE_FS_SPEC(FromString_3,uint3)

	MD_DEFINE_FS_SPEC(FromString_2,float2)
	MD_DEFINE_FS_SPEC(FromString_2,int2)
	MD_DEFINE_FS_SPEC(FromString_2,uint2)

#undef MD_DEFINE_FS_SPEC


}