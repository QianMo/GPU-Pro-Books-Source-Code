#ifndef PROVIDERS_CACHEMAP_H_INCLUDED
#define PROVIDERS_CACHEMAP_H_INCLUDED

#include "Forw.h"

#include "ExportDefs.h"

#define MD_NAMESPACE CacheMapNS
#include "ConfigurableImpl.h"


namespace Mod
{

	class CacheMap : public CacheMapNS::ConfigurableImpl<CacheMapConfig>
	{
		// types
	public:

		// some standard drilling folks suggest we can rely on order of elements with same multimap key.
		// paranoia however suggests we shouldn't, thus we have to store extra index

		struct NameEntry
		{
			String name;
			UINT32 index;
		};

		typedef Types2< UINT64, NameEntry > :: MultiMap FileNameHelperMap;

		// construction/ destruction
	public:
		explicit CacheMap( const CacheMapConfig& cfg );
		~CacheMap();

		// manipulation/ access
	public:
		String	GetFileName( const String& key, bool& oFirstTimeEntry );
		void	Save();

		// data
	private:
		FileNameHelperMap	mFileNameHelperMap;
		bool				mModified;
	};
}

#endif