#ifndef PROVIDERS_ANIMATIONTRACKIMPL_H_INCLUDED
#define PROVIDERS_ANIMATIONTRACKIMPL_H_INCLUDED

#include "Math/Src/Types.h"

namespace Mod
{
	template <typename TC, typename RC, typename T=float> // position/ rotation curves
	class AnimationTrackImpl
	{
		// types
	public:
		typedef T Time;
		typedef TC TCurve;
		typedef RC RCurve;

		// construction /destruction
	public:
		AnimationTrackImpl(const TC& pc, const RC& rc);
	protected:
		~AnimationTrackImpl();

		  // manipulation /access
	public:

		Math::float4 EvalRot(Time time) const;
		Math::float3 EvalPos(Time time) const;

		Math::float3x4 Evaluate(Time time) const;
		Math::float3x4 Evaluate(Time time, float weight) const;

		Time GetStartTime() const;
		Time GetEndTime() const;

		const TCurve& GetPosCurve() const;
		const RCurve& GetRotCurve() const;

		void Transform( const Math::float3x4& transform );

		// data
	private:
		TCurve	mPosCurve;
		RCurve	mRotCurve;
	};

	//------------------------------------------------------------------------






}

#endif