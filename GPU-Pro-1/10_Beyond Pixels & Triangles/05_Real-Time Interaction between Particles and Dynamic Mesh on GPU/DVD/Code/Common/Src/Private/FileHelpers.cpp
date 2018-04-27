#include "Precompiled.h"

#include "Math/Src/BBox.h"
#include "Math/Src/Operations.h"

#include "WrapSys/Src/System.h"

#include "WrapSys/Src/FileInfo.h"

#include "WrapSys/Src/FileConfig.h"
#include "WrapSys/Src/File.h"

#include "SplitPath.h"

#include "FileHelpers.h"

namespace Mod
{

	RFilePtr CreateRFile( const String& fileName )
	{
		RFileConfig rcfg;
		rcfg.fileName = fileName;

		return ToRFilePtr( System::Single().CreateFile( rcfg ) );
	}

	//------------------------------------------------------------------------

	WFilePtr CreateFileAndPath( const String& fileName )
	{
		System& sys = System::Single();

		SplitPath sp( fileName );

		const String& dir = sp.Dir();
		size_t off = 0, e = 0;

		while( ( e = dir.find_first_of( L"\\/", off ) ) != String::npos )
		{
			// skip possible first slash (when the pass is not relative)
			if( e )
			{
				String currPath = sp.Drive()  + String( dir.begin(), dir.begin() + e );

				if( !sys.DirExists( currPath ) )
				{
					sys.CreateDir( currPath );
				}
			}
			off = e + 1;
		}

		WFileConfig wcfg( true );
		wcfg.fileName		= fileName;

		return ToWFilePtr( sys.CreateFile( wcfg ) );		
	}

	//------------------------------------------------------------------------

	namespace
	{
		void finalize( String& str )
		{
			if( !str.empty() )
			{
				bool reverse = false;

				if( str.find_first_of( '/' ) != String::npos )
					reverse = true;

				String::value_type ch = *(str.end()-1);
				if( ch != '\\' && ch != '/' )
					str.append( 1, reverse ? '/' : '\\' );
			}
		}
	}

	String GetRelativePath( const String& path )
	{
		return GetRelativePath( System::Single().GetCurrentPath(), path );
	}

	//------------------------------------------------------------------------

	String GetRelativePath( const String& base, const String& path )
	{
		System& sys = System::Single();

		String basePath( sys.GetFullPath( base ) );
		ToUpper( basePath );
		finalize( basePath );

		String fullPath( sys.GetFullPath( path ) );
		ToUpper( fullPath );

		SplitPath sp_base( basePath );
		SplitPath sp_full( fullPath );

		if( sp_base.Drive() != sp_full.Drive() )
			return String();

		size_t i = 0;

		String folderBase;
		String folderFull;
		for(;;)
		{
			size_t newPosBase = basePath.find_first_of( L"/\\", i );
			size_t newPosFull = fullPath.find_first_of( L"/\\", i );

			if( newPosBase == String::npos || newPosFull == String::npos )
				break;

			if( String( basePath.begin() + i, basePath.begin() + newPosBase ) != 
				String( fullPath.begin() + i, fullPath.begin() + newPosFull ) )
				break;

			i = newPosBase + 1;
		}

		size_t start_pos = i;

		String result;

		i = basePath.find_first_of( L"/\\", i );

		while ( i != String::npos )
		{
			result += L"..\\";
			i = basePath.find_first_of( L"/\\", i+1 );
		}

		result += String( fullPath.begin() + start_pos, fullPath.end() );

		return result;
	}

	//------------------------------------------------------------------------

	bool ArePathesEqual( const String& path1, const String& path2 )
	{
		System& sys = System::Single();
		String p1 = sys.GetFullPath( path1 );
		String p2 = sys.GetFullPath( path2 );

		ToUpper( p1 );
		ToUpper( p2 );

		return p1 == p2;
	}

	//------------------------------------------------------------------------

	FileStamp GetFileStamp( const String& fileName )
	{
		return System::Single().GetFileInfo( fileName ).stamp;
	}

	//------------------------------------------------------------------------

	FileStamp GetFileStamp( const RFilePtr& file )
	{
		return System::Single().GetFileInfo( file ).stamp;
	}

	//------------------------------------------------------------------------

	FileStamp GetFileStamp( const WFilePtr& file )
	{
		return System::Single().GetFileInfo( file ).stamp;
	}

	//------------------------------------------------------------------------

	Strings
	GatherFileNames( const String& path, const String& wildCard )
	{
		return System::Single().GatherFileNames( path, wildCard, true );
	}

	//------------------------------------------------------------------------

	void ReadBytes( const String& fileName, Bytes& oBytes )
	{
		RFilePtr rfile = CreateRFile( fileName );
		rfile->Read( oBytes );
	}

	//------------------------------------------------------------------------

	Math::BBox ReadBBoxFromFile( const RFilePtr& file )
	{
		using namespace Math;

		float3 bmin, bmax;

		file->Read( bmin );
		file->Read( bmax );

		return BBox( bmin, bmax );
	}

	//------------------------------------------------------------------------

	namespace
	{

		size_t estimate( const WCHAR* fmt, va_list args )
		{
			return _vscwprintf( fmt, args );
		}
	
		size_t estimate( const CHAR* fmt, va_list args )
		{
			return _vscprintf( fmt, args );
		}

		void vprint( WCHAR* buf, size_t size, const WCHAR* fmt, va_list args )
		{
			vswprintf( buf, size, fmt, args );
		}

		void vprint( CHAR* buf, size_t size, const CHAR* fmt, va_list args )
		{
			size;
			vsprintf( buf, fmt, args );
		}

		template <typename T>
		void FilePrintfImpl( const WFilePtr& file, const T* fmt, va_list args )
		{
			Types < T > :: Vec res;

			size_t len = estimate( fmt, args ) + 1;

			res.resize( len );

			vprint( &res[0], res.size(), fmt, args );

			res.resize( res.size() - 1 );

			if( !res.empty() )
			{
				FileWriteContainer( file, res );
			}
		}
	}

	//------------------------------------------------------------------------

	void FilePrintf( const WFilePtr& file, const WCHAR* fmt, ... )
	{
		va_list args;
		va_start( args, fmt );

		FilePrintfImpl( file, fmt, args );
	}

	//------------------------------------------------------------------------

	void FileAnsiPrintf( const WFilePtr& file, const CHAR* fmt, ... )
	{
		va_list args;
		va_start( args, fmt );

		FilePrintfImpl( file, fmt, args );
	}

	//------------------------------------------------------------------------

	void FileWrite( const WFilePtr& file, const String& str )
	{
		UINT16 length = (UINT16)str.size();
		file->Write( length );
		FileWriteContainer( file, str );
	}

	//------------------------------------------------------------------------

	void FileRead( const RFilePtr& file, String& oStr )
	{
		UINT16 length;
		file->Read( length );
		FileReadContainer( file, length, oStr );
	}


}