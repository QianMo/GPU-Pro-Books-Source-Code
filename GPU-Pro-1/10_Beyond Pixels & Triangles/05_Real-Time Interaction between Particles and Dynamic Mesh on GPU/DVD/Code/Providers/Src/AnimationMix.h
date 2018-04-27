#ifndef PROVIDERS_ANIMATIONMIX_H_INCLUDED
#define PROVIDERS_ANIMATIONMIX_H_INCLUDED

#include "ExportDefs.h"

#include "AnimationInfo.h"

namespace Mod
{
	class AnimationMix
	{
		// types
	public:
		struct Entry
		{
			UINT32 idx; // index on skeleton
			float t;
			float weight;

			AnimationInfo info;
		};

		typedef Types< Entry > :: Vec Entries;

		// construction/ destruction
	public:
		EXP_IMP AnimationMix();
		EXP_IMP ~AnimationMix();

		// manipulation/ access
	public:
		EXP_IMP void EnableAnim( UINT32 idx, const AnimationInfo& info );
		EXP_IMP void DisableAnim( UINT32 idx );
		EXP_IMP void SetAnimWeight( UINT32 idx, float weight );
		EXP_IMP bool IsAnimEnabled( UINT32 idx ) const;
		EXP_IMP AnimationInfo& GetModifiableAnimationInfo( UINT32 idx );

		EXP_IMP void Update( float dt );

		EXP_IMP const Entries& GetEntries() const;

		// data
	private:
		Entries mEntries;
	};
}

#endif