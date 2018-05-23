/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_ANIMATEDHOST
#define BE_ENTITYSYSTEM_ANIMATEDHOST

#include "beEntitySystem.h"
#include "beAnimated.h"
#include <vector>

namespace beEntitySystem
{

/// Animated interface.
class AnimatedHost : public Animated
{
private:
	typedef std::vector<Animated*> animated_vector;
	animated_vector m_animate;

public:
	/// Constructor.
	BE_ENTITYSYSTEM_API AnimatedHost();
	/// Destructor.
	BE_ENTITYSYSTEM_API ~AnimatedHost();

	/// Steps the animation.
	BE_ENTITYSYSTEM_API void Step(float timeStep);

	/// Adds an animated controller.
	BE_ENTITYSYSTEM_API void AddAnimated(Animated *animated);
	/// Removes an animated controller.
	BE_ENTITYSYSTEM_API void RemoveAnimated(Animated *animated);
};

} // namespace

#endif