#ifndef COMMON_FILEHELPERS_H_INCLUDED
#define COMMON_FILEHELPERS_H_INCLUDED

#include "Math/Src/Forw.h"
#include "WrapSys/Src/Forw.h"

#include "WrapSys/Src/File.h"

namespace Mod
{
	RFilePtr	CreateRFile( const String& fileName );
	WFilePtr	CreateFileAndPath( const String& fileName );
	String		GetRelativePath( const String& path );
	String		GetRelativePath( const String& base, const String& path );
	bool		ArePathesEqual( const String& path1, const String& path2 );

	FileStamp	GetFileStamp( const String& fileName );
	FileStamp	GetFileStamp( const RFilePtr& file );
	FileStamp	GetFileStamp( const WFilePtr& file );

	Strings		GatherFileNames( const String& path, const String& wildCard );

	void		ReadBytes( const String& fileName, Bytes& oBytes );
	Math::BBox	ReadBBoxFromFile( const RFilePtr& file );
	void		FilePrintf( const WFilePtr& file, const WCHAR* fmt, ... );
	void		FileAnsiPrintf( const WFilePtr& file, const CHAR* fmt, ... );
	void		FileRead( const RFilePtr& file, String& oStr );
	void		FileWrite( const WFilePtr& file, const String& str );

	//------------------------------------------------------------------------

	template< typename T >
	void		FileWrite( const WFilePtr& file, const T& val )
	{
		file->Write( val );
	}

	//------------------------------------------------------------------------

	template< typename T >
	void		FileRead( const RFilePtr& file, T& val )
	{
		file->Read( val );
	}

	//------------------------------------------------------------------------

	template <typename C>
	void		FileWriteContainer( const WFilePtr& file, const C& c )
	{
		for( C::const_iterator i = c.begin(), e = c.end(); i != e; ++i )
		{
			FileWrite( file, *i );
		}
	}

	//------------------------------------------------------------------------

	template <typename C>
	void		FileReadContainer( const RFilePtr& file, UINT32 size, C& oC )
	{
		oC.resize( size );
		for( C::iterator i = oC.begin(), e = oC.end(); i != e; ++i )
		{
			FileRead( file, *i );
		}
	}

	//------------------------------------------------------------------------

	template <typename M>
	void FileWriteMap( const WFilePtr& file, const M& m )
	{
		UINT64 size = m.size();
		file->Write( size );

		for( M::const_iterator i = m.begin(), e = m.end(); i != e; ++i )
		{
			FileWrite( file, i->first );
			FileWrite( file, i->second );
		}	
	}

	//------------------------------------------------------------------------

	template <typename M>
	void FileReadMap( const RFilePtr& file, M& m )
	{
		UINT64 size;
		file->Read( size );

		for( UINT64 i = 0; i < size; i ++ )
		{
			typename M::key_type	key;
			typename M::mapped_type val;

			FileRead( file, key	);
			FileRead( file, val	);

			m[ key ] = val;
		}	
	}	
}

#endif