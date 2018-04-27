#include "Precompiled.h"

#include "Common/Src/FileHelpers.h"

#include "WrapSys/Src/FileStamp.h"
#include "WrapSys/Src/FileConfig.h"
#include "WrapSys/Src/File.h"
#include "WrapSys/Src/System.h"

#include "Wrap3D/Src/DeviceConfig.h"
#include "Wrap3D/Src/Device.h"

#include "Wrap3D/Src/EffectConfig.h"
#include "Wrap3D/Src/Effect.h"

#include "CacheMap.h"
#include "EffectIncludeProvider.h"

#include "Providers.h"

#include "EffectProviderConfigBase.h"

#include "EffectDefine.h"
#include "EffectKey.h"

#include "EffectProviderCommon.h"

namespace Mod
{

	namespace
	{
		String ConstructStringKey( const EffectKey& key );
		bool SetCachedCode( EffectConfigBase& ecfg, const String& filePath, const String& cacheFilePath, bool allowCreateCache, bool forceCache );
	}

	void FillEffectConfigBaseWithCompiledCode( const EffectCompiler& cpler, EffectProviderConfigBase pcfg, const EffectKey& effkey, EffectConfigBase& oCfg )
	{
		DevicePtr dev = pcfg.dev;

		EffectKey key = effkey;

		const DeviceConfig::EffDefs& effDefs = dev->GetConfig().PLATFORM_EFFECT_DEFINES;

		for( size_t i = 0, e = effDefs.size(); i < e; i ++ )
		{
			EffectDefine edef;
			edef.name	= effDefs[i].name;
			edef.val	= effDefs[i].val;
			key.defines.insert( edef );
		}

		const String& fileName = key.file;

		bool firstTimeEntry;
		const String& cacheFileName = Providers::Single().GetEffectCacheMap()->GetFileName( ConstructStringKey( key ), firstTimeEntry );

		const String& cacheFilePath = pcfg.cachePath + cacheFileName + pcfg.cachedExtension;
		const String& filePath = pcfg.path + fileName + pcfg.extension;

		bool forceRecompile = !pcfg.forceCache && Providers::Single().GetEffectIncludeProv()->AreIncludesModified();

		if( forceRecompile || firstTimeEntry || !SetCachedCode( oCfg, filePath, cacheFilePath, pcfg.autoCreateCache, pcfg.forceCache ) )
		{
			String errors;

			RFileConfig fcfg;
			fcfg.fileName = filePath;
			RFilePtr file = ToRFilePtr( System::Single().CreateFile( fcfg ) );

			Bytes bytes;
			file->Read( bytes );

			if( !cpler.Compile( bytes, key.defines, oCfg.code, errors) )
				MD_THROW( L"Couldn't compile effect " + fileName + L" errors:" + errors );
			oCfg.compiled = true;

			if( pcfg.autoCreateCache )
			{
				WFilePtr ofile = CreateFileAndPath( cacheFilePath );
				ofile->Write( GetFileStamp( file ) );
				ofile->Write( oCfg.code );
			}
		}
	}

	namespace
	{

		String ConstructStringKey( const EffectKey& key )
		{
			String res( key.file );

			res += key.poolFile;

			for( EffectKey::Defines::const_iterator i = key.defines.begin(), e = key.defines.end(); i != e; ++i )
			{
				const EffectDefine& def = *i;
				res += ToString( def.name );
				res += ToString( def.val );
			}

			return res;
		}

		bool SetCachedCode( EffectConfigBase& ecfg, const String& filePath, const String& cacheFilePath, bool allowCreateCache, bool forceCache )
		{
			if( System::Single().FileExists( cacheFilePath ) )
			{
				RFileConfig fcfg;
				fcfg.fileName = cacheFilePath;
				RFilePtr codeFile = ToRFilePtr( System::Single().CreateFile( fcfg ) );

				FileStamp currStamp;
				codeFile->Read( currStamp );

				FileStamp stamp = currStamp;

				if( System::Single().FileExists( filePath ) )
				{
					stamp = GetFileStamp( filePath );
				}

				if( currStamp != stamp && !forceCache )
				{
					if( allowCreateCache )
					{
						codeFile.reset();
						System::Single().DeleteFile( cacheFilePath );
					}
					return false;
				}

				codeFile->Read( ecfg.code );
				ecfg.compiled = true;

				return true;
			}
			return false;
		}
	}
}