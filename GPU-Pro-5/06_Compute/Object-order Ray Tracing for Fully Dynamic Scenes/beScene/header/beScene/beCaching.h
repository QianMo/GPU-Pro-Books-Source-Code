/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_CACHING
#define BE_SCENE_CACHING

#include "beScene.h"

namespace beScene
{

/// Supports invalidation of internally cached data.
class LEAN_INTERFACE Caching
{
	LEAN_INTERFACE_BEHAVIOR(Caching)

public:
	/// Invalidates all caches.
	virtual void InvalidateCaches() = 0;
};

} // namespace

#endif