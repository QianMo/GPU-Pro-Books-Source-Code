#include "Precompiled.h"

#include "Math/Src/Operations.h"

#include "V3Curve.h"
#include "QRCurve.h"

#include "AnimationTrackImpl.h"

namespace Mod
{
	using namespace Math;

	//------------------------------------------------------------------------

	template <typename TC, typename RC, typename T>
	AnimationTrackImpl<TC, RC, T>::AnimationTrackImpl(const TC& pc, const RC& rc):
	mPosCurve( pc ),
	mRotCurve( rc )
	{
		
	}

	//------------------------------------------------------------------------

	template <typename TC, typename RC, typename T>
	AnimationTrackImpl<TC, RC, T>::~AnimationTrackImpl()
	{

	}

	//------------------------------------------------------------------------

	template <typename TC, typename RC, typename T>
	float4
	AnimationTrackImpl<TC, RC, T>::EvalRot(Time time) const
	{
		return mRotCurve.Evaluate(time);
	}

	//------------------------------------------------------------------------

	template <typename TC, typename RC, typename T>
	float3
	AnimationTrackImpl<TC, RC, T>::EvalPos(Time time) const
	{
		return mPosCurve.Evaluate(time);
	}

	//------------------------------------------------------------------------

	template <typename TC, typename RC, typename T>
	float3x4
	AnimationTrackImpl<TC, RC, T>::Evaluate(Time time) const
	{
		float3	pos		= EvalPos(time);
		float4	rot		= EvalRot(time);

		return m3x4Transform( float3(1,1,1), rot, pos );
	}

	//------------------------------------------------------------------------

	template <typename TC, typename RC, typename T>
	float3x4
	AnimationTrackImpl<TC, RC, T>::Evaluate( Time time, float weight ) const
	{
		float3	pos		= EvalPos( time ) * weight;
		float4	rot		= EvalRot(time);

		rot = normalize( lerp( IDENTITY_QUAT, rot, weight ) );

		return m3x4Transform( float3(1,1,1), rot, pos );
	}

	//------------------------------------------------------------------------

	template <typename TC, typename RC, typename T>
	T
	AnimationTrackImpl<TC, RC, T>::GetStartTime() const
	{
		Time pt = mPosCurve.GetStartTime();
		Time rt = mRotCurve.GetStartTime();
		return std::min(pt, rt);
	}

	//------------------------------------------------------------------------

	template <typename TC, typename RC, typename T>
	T
	AnimationTrackImpl<TC, RC, T>::GetEndTime() const
	{
		Time pt = mPosCurve.GetEndTime();
		Time rt = mRotCurve.GetEndTime();
		return std::max(pt, rt);
	}

	//------------------------------------------------------------------------

	template <typename TC, typename RC, typename T>
	const TC&
	AnimationTrackImpl<TC, RC, T>::GetPosCurve() const
	{
		return mPosCurve;
	}

	//------------------------------------------------------------------------

	template <typename TC, typename RC, typename T>
	const RC&
	AnimationTrackImpl<TC, RC, T>::GetRotCurve() const
	{
		return mRotCurve;
	}

	//------------------------------------------------------------------------

	template <typename TC, typename RC, typename T>
	void
	AnimationTrackImpl<TC, RC, T>::Transform( const Math::float3x4& transform )
	{
		mRotCurve.Transform( transform );
		mPosCurve.Transform( transform );
	}

	//------------------------------------------------------------------------

	template class AnimationTrackImpl< V3Curve, QRCurve >;


}