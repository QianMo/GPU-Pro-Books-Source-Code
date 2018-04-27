#include "Precompiled.h"

#include "WrapSys/Src/FileStamp.h"
#include "WrapSys/Src/System.h"

#include "Common/Src/FileHelpers.h"

#include "EffectIncludeProviderConfig.h"
#include "EffectIncludeProvider.h"

#define MD_NAMESPACE EffectIncludeProviderNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{

	template class EffectIncludeProviderNS::ConfigurableImpl<EffectIncludeProviderConfig>;

	//------------------------------------------------------------------------

	namespace
	{
		String getCacheFileName( const EffectIncludeProviderConfig& cfg )
		{
			return cfg.cacheFilePath + L"stamps.dat";
		}
	}

	EXP_IMP
	EffectIncludeProvider::EffectIncludeProvider( const EffectIncludeProviderConfig& cfg ) :
	Parent( cfg ),
	mIncludesModified( false )
	{
		const String& cacheFileName = getCacheFileName( cfg );

		if( System::Single().FileExists( cacheFileName ) )
		{
			RFilePtr file = CreateRFile( cacheFileName );
			FileReadMap( file, mIncludeStampMap );
		}
		else
			mIncludesModified = true;

		Strings files = System::Single().GatherFileNames( cfg.path, L"*" );

		for( size_t i = 0, e = files.size(); i < e; i ++ )
		{
			const String& file = files[i];
			const String& filePath = cfg.path + file;
			FileStamp stamp = GetFileStamp( filePath );

			FileStampMap::iterator found = mIncludeStampMap.find( file );

			if( found != mIncludeStampMap.end() )
			{
				if( found->second != stamp )
				{
					found->second = stamp;
					mIncludesModified = true;
				}
			}
			else
				mIncludeStampMap[ file ] = stamp;
		}
	}

	//------------------------------------------------------------------------

	EXP_IMP
	EffectIncludeProvider::~EffectIncludeProvider()
	{
		if( mIncludesModified )
		{
			const ConfigType& cfg = GetConfig();
			WFilePtr file = CreateFileAndPath( getCacheFileName( cfg ) );

			FileWriteMap( file, mIncludeStampMap );
		}
	}

	//------------------------------------------------------------------------

	bool
	EffectIncludeProvider::AreIncludesModified() const
	{
		return mIncludesModified;
	}

	//------------------------------------------------------------------------
	
	EffectIncludeProvider::ItemTypePtr
	EffectIncludeProvider::CreateItemImpl( const KeyType& key )
	{
		BytesPtr bytes( new Bytes );

		const String& pathFName = GetConfig().path + key; 

		if( System::Single().FileExists( pathFName ) )
		{
			ReadBytes( pathFName, *bytes );
		}
		else
			ReadBytes( key, *bytes );

		return bytes;
	}

}