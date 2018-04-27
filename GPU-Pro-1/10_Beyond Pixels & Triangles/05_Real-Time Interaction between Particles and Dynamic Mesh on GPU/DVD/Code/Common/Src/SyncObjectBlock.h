#ifndef COMMON_SYNCOBJECTBLOCK_H_INCLUDED
#define COMMON_SYNCOBJECTBLOCK_H_INCLUDED

#include "WrapSys/Src/Forw.h"

namespace Mod
{
	template <typename T>
	class SyncObjectBlock
	{
		// construction/ destruction
	public:
		explicit SyncObjectBlock( const T& val );
		~SyncObjectBlock();

		// resturict
	private:
		SyncObjectBlock( const SyncObjectBlock& );
		void operator = ( const SyncObjectBlock& );

		// data
	private:
		const T& mVal;
	};

	//------------------------------------------------------------------------
	/*explicit*/

	template <typename T>
	SyncObjectBlock<T>::SyncObjectBlock( const T& val ) :
	mVal( val )
	{
		mVal->Capture();
	}

	//------------------------------------------------------------------------

	template <typename T>
	SyncObjectBlock<T>::~SyncObjectBlock()
	{
		mVal->Release();
	}

	//------------------------------------------------------------------------

	typedef SyncObjectBlock< MutexPtr >				MutexBlock;
	typedef SyncObjectBlock< CriticalSectionPtr >	CriticalSectionBlock;

}

#endif

