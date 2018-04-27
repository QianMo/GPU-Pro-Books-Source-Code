#include "Precompiled.h"

#include "V3Curve.h"
#include "QRCurve.h"

#include "Curve.h"

namespace Mod
{
	template <typename Child, typename T, typename Time>
	Curve<Child, T, Time>::Curve() :
	mLastKeyTime(0),
	mFirstKeyTime(0),
	mRange(0),
	mNumKeys(0)
	{

	}

	//------------------------------------------------------------------------

	template <typename Child, typename T, typename Time>
	Curve<Child, T, Time>::~Curve()
	{

	}

	//------------------------------------------------------------------------

	template <typename Child, typename T, typename Time>
	void
	Curve<Child, T, Time>::AddKey(const T& value, Time time)
	{
		MD_FERROR_ON_FALSE( time >= 0.f );

		Keys::iterator end = mKeys.end();
		Keys::iterator found = std::find_if(mKeys.begin(), end, FindAfter(time));

		if(found == end)
		{
			mKeys.push_back(Key(value,time));
			mLastKeyTime = time;
		}
		else
			mKeys.insert(found, Key(value,time));

		mFirstKeyTime = mKeys.front().time;

		mRange = mLastKeyTime - mFirstKeyTime;

		mNumKeys++;
	}

	//------------------------------------------------------------------------

	template <typename Child, typename T, typename Time>
	T
	Curve<Child, T, Time>::Evaluate( Time t ) const
	{
		// aimed towards lotsa keys and uniform in-time distribution (what we usually have)
		MD_ASSERT( !mKeys.empty() );

		// find the key first
		Time normTime = mRange ? (t-mFirstKeyTime)/mRange : 0;

		if(normTime >= 1)
			return mKeys.back().value;
		if(normTime <= 0)
			return mKeys.front().value;	// times are required to be >= 0

		UINT32 idx = UINT32(normTime * (mNumKeys - 1 ));

		while(mKeys[idx].time > t && idx)
			idx --;

		while(mKeys[idx].time < t && idx < mNumKeys-1)
			idx ++ ;

		if(!idx)
			return mKeys.front().value;

		MD_ASSERT( idx < mNumKeys );

		const Key &k1 = mKeys[idx];
		const Key &k2 = mKeys[idx-1];
		return Child::Interpolate(k1.value, k2.value, (t-k1.time)/(k2.time-k1.time) );

	}

	//------------------------------------------------------------------------

	template <typename Child, typename T, typename Time>
	Time
	Curve<Child, T, Time>::GetStartTime() const
	{
		return mFirstKeyTime;
	}

	//------------------------------------------------------------------------

	template <typename Child, typename T, typename Time>
	Time
	Curve<Child, T, Time>::GetEndTime() const
	{
		return mLastKeyTime;
	}

	//------------------------------------------------------------------------

	template <typename Child, typename T, typename Time>
	typename const Curve<Child, T, Time>::Keys&
	Curve<Child, T, Time>::GetKeys() const
	{
		return mKeys;
	}

	//------------------------------------------------------------------------

	template <typename Child, typename T, typename Time>
	void
	Curve<Child, T, Time>::Transform( const Math::float3x4& trans )
	{
		Math::float3x4 mtrans = Child::ModifyTransformMatrix( trans );

		for( size_t i = 0, e = mKeys.size(); i < e; i ++ )
		{
			Child::TransformKey( mKeys[i], mtrans );
		}
	}

	//------------------------------------------------------------------------

	template class Curve< V3Curve, V3Curve::ItemType >;
	template class Curve< QRCurve, QRCurve::ItemType >;
}