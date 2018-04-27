#include "Precompiled.h"

#include "AnimationMix.h"

namespace Mod
{
	EXP_IMP
	AnimationMix::AnimationMix()
	{

	}

	//------------------------------------------------------------------------

	EXP_IMP
	AnimationMix::~AnimationMix()
	{

	}

	//------------------------------------------------------------------------

	namespace
	{
		struct CheckEntryIdx
		{
			CheckEntryIdx( UINT32 a_idx ) : 
			idx( a_idx )
			{
			}

			bool operator() ( const AnimationMix::Entry& e )
			{
				return e.idx == idx;
			}

			UINT32 idx;
		};
	}

	EXP_IMP
	void
	AnimationMix::EnableAnim( UINT32 idx, const AnimationInfo& info )
	{
		Entries::iterator found = std::find_if( mEntries.begin(), mEntries.end(), CheckEntryIdx(idx) );

		Entry e;
		e.idx		= idx;
		e.t			= 0.f;
		e.weight	= 1.f;
		e.info		= info;

		if(  found == mEntries.end() )
		{
			mEntries.push_back( e );
		}
		else
		{
			(*found) = e;
		}
	}

	//------------------------------------------------------------------------

	EXP_IMP
	void
	AnimationMix::DisableAnim( UINT32 idx )
	{
		Entries::iterator found = std::find_if( mEntries.begin(), mEntries.end(), CheckEntryIdx( idx ) );

		MD_FERROR_ON_TRUE( found == mEntries.end() );

		mEntries.erase( found );
	}

	//------------------------------------------------------------------------

	EXP_IMP
	void
	AnimationMix::SetAnimWeight( UINT32 idx, float weight )
	{
		Entries::iterator found = std::find_if( mEntries.begin(), mEntries.end(), CheckEntryIdx( idx ) );

		MD_FERROR_ON_TRUE( found == mEntries.end() );

		(*found).weight = weight;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	bool
	AnimationMix::IsAnimEnabled( UINT32 idx ) const
	{
		return std::find_if( mEntries.begin(), mEntries.end(), CheckEntryIdx( idx ) ) != mEntries.end();
	}

	//------------------------------------------------------------------------

	EXP_IMP
	AnimationInfo&
	AnimationMix::GetModifiableAnimationInfo( UINT32 idx )
	{
		Entries::iterator found = std::find_if( mEntries.begin(), mEntries.end(), CheckEntryIdx( idx ) );

		MD_FERROR_ON_TRUE( found == mEntries.end() );

		return ( *found ).info;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	void
	AnimationMix::Update( float dt )
	{
		for( Entries::iterator i = mEntries.begin(), end = mEntries.end(); i != end; ++i )
		{
			Entry& e = *i;
			
			e.t += dt * e.info.speed;

			if( e.t > e.info.length )
			{
				e.t = fmodf( e.t, e.info.length );
			}
		}
	}

	//------------------------------------------------------------------------

	const
	AnimationMix::Entries&
	AnimationMix::GetEntries() const
	{
		return mEntries;
	}

}
