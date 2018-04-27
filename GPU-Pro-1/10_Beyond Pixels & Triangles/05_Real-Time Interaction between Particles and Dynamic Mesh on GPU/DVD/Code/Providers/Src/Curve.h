#ifndef PROVIDERS_CURVE_H_INCLUDED
#define PROVIDERS_CURVE_H_INCLUDED

namespace Mod
{
	template <typename Child, typename T, typename Time = float>
	class Curve
	{
		// types
	public:

		typedef Curve Parent;
		typedef Time Time;
		typedef T ItemType;

		struct Key
		{
			Key( const T& value, Time time): value(value), time(time) {}
			T		value;
			Time	time;
		};

		// helps us in our key seeking tasks
		// TODO : binary search!
		struct FindAfter
		{
			FindAfter(Time time):time(time) 
			{}
			bool operator() (const Key& key)
			{	return key.time > time;	}
			Time time;
		};

		typedef std::vector<Key> Keys;

		// construction/ destruction
	public:
		Curve();
		~Curve();

		  // manipulation/access
	public:
		void		AddKey( const T& value, Time time );

		T			Evaluate( Time t ) const;

		Time		GetStartTime() const;
		Time		GetEndTime() const;
		const Keys&	GetKeys() const;

		void		Transform( const Math::float3x4& trans );

		// data
	private:
		Keys	mKeys;
		Time	mLastKeyTime;
		Time	mFirstKeyTime;
		Time	mRange;
		UINT32	mNumKeys;
	};
}

#endif