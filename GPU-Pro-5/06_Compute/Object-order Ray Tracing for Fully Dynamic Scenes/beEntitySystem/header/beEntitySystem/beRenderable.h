/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_RENDERABLE
#define BE_ENTITYSYSTEM_RENDERABLE

#include "beEntitySystem.h"

namespace beEntitySystem
{

/// Renderable interface.
class LEAN_INTERFACE Renderable
{
public:
	/// Renders renderable content.
	virtual void Render() = 0;
};

} // namespace

#endif