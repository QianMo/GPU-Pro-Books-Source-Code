#ifndef PROVIDERS_ANIMATIONINFO_H_INCLUDED
#define PROVIDERS_ANIMATIONINFO_H_INCLUDED

namespace Mod
{
	struct AnimationInfo
	{
		AnimationInfo();

		float	length;
		float	speed;
		bool	loop;
	};

	//------------------------------------------------------------------------

	inline
	AnimationInfo::AnimationInfo() :
	length( 0 ),
	speed( 1.f ),
	loop( false )
	{

	}
}

#endif