#include "Precompiled.h"

#include "WrapSys/Src/System.h"

#include "Common/Src/FileHelpers.h"
#include "Common/Src/StringUtils.h"
#include "Common/Src/SplitPath.h"

#include "CacheMapConfig.h"
#include "CacheMap.h"

#define MD_NAMESPACE CacheMapNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	template class CacheMapNS::ConfigurableImpl<CacheMapConfig>;

	//------------------------------------------------------------------------

	namespace
	{
		void SaveToFile( const String& fileName, const CacheMap::FileNameHelperMap& map );
		void LoadFromFile( const String& fileName, CacheMap::FileNameHelperMap& oMap );
		UINT64 ExtractKey( const String& key );
		String ExtractName( const CacheMap::NameEntry& entry );
	}

	//------------------------------------------------------------------------

	CacheMap::CacheMap( const CacheMapConfig& cfg ) : 
	Parent( cfg ),
	mModified( false )
	{
		LoadFromFile( cfg.fileName, mFileNameHelperMap );
	}

	//------------------------------------------------------------------------

	CacheMap::~CacheMap()
	{
		Save();
	}

	//------------------------------------------------------------------------

	String
	CacheMap::GetFileName( const String& key, bool& oFirstTimeEntry )
	{
		String ukey( key );
		ToUpper( ukey );

		const ConfigType& cfg = GetConfig();

		oFirstTimeEntry = false;

		UINT64 bkey = ExtractKey( ukey );

		size_t count = mFileNameHelperMap.count( bkey );

		if( size_t i = count )
		{
			FileNameHelperMap::const_iterator found = mFileNameHelperMap.find( bkey );

			do
			{
				if( found->second.name == ukey )
					return cfg.cachePath + ExtractName( found->second );

				found++;
			}
			while( --i );
		}

		mModified = true;


		NameEntry e;
		e.index	= UINT32(count);
		e.name	= ukey;

		oFirstTimeEntry = true;
		mFileNameHelperMap.insert( FileNameHelperMap::value_type( bkey, e ) );

		return cfg.cachePath + ExtractName( e );
	}

	//------------------------------------------------------------------------

	void
	CacheMap::Save()
	{
		if( mModified )
		{
			SaveToFile( GetConfig().fileName, mFileNameHelperMap );
			mModified = false;
		}
	}

	//------------------------------------------------------------------------

	namespace
	{

		void SaveToFile( const String& fileName, const CacheMap::FileNameHelperMap& map )
		{
			WFilePtr file = CreateFileAndPath( fileName );

			UINT32 size = (UINT32)map.size();
			file->Write( size );

			for( CacheMap::FileNameHelperMap::const_iterator i = map.begin(), e = map.end(); i != e; ++i )
			{
				file->Write( i->first );
				FileWrite( file, i->second.name );
				file->Write( i->second.index );
			}
		}

		void LoadFromFile( const String& fileName, CacheMap::FileNameHelperMap& oMap )
		{
			typedef CacheMap::FileNameHelperMap Map;

			if( System::Single().FileExists( fileName ) )
			{
				RFilePtr file = CreateRFile( fileName );

				UINT32 size;
				file->Read( size );

				for( UINT32 i = 0; i < size; i ++ )
				{
					Map::key_type key;
					Map::mapped_type e;
					file->Read( key );
					FileRead( file, e.name );
					file->Read( e.index );

					oMap.insert( Map::value_type( key, e ) );
				}				
			}
		}

		UINT64 ExtractKey( const String& key )
		{
			UINT64 result( 0 );

			size_t size = key.size();

			if( size )
			{
				result |= key[0];

				if( size > 1 )
				{
					result |= key[1] << 16;
					if( size > 2 )
					{
						result |= (UINT64)key[2] << 32ull;
						if( size > 3 )
						{
							result |= (UINT64)key[3] << 48ull;
						}
					}
				}
			}

			return result;
		}

		String ExtractName( const CacheMap::NameEntry& entry )
		{
			WCHAR name[5] = {};
			
			SplitPath sp( entry.name );

			wcsncpy( name, sp.FName().c_str(), sizeof( name ) / sizeof( name[0] )- 1 );

			return sp.Drive() + sp.Dir() + name + AsString( (int)entry.index );
		}
	}

}