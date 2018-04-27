#ifndef PROVIDERS_ANIMATIONTRACK_H_INCLUDED
#define PROVIDERS_ANIMATIONTRACK_H_INCLUDED

#include "QRCurve.h"
#include "V3Curve.h"

#include "AnimationTrackImpl.h"

namespace Mod
{
	class AnimationTrack : public AnimationTrackImpl< V3Curve, QRCurve >
	{
		// types
	public:
		typedef AnimationTrackImpl< V3Curve, QRCurve > Base;

		// construction/ destruction
	public:
		AnimationTrack(const V3Curve& pc, const QRCurve& rc);
		~AnimationTrack();


	};

}

#endif