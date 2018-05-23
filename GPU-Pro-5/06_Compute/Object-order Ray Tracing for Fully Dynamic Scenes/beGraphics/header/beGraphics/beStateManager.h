/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_STATE_MANAGER
#define BE_GRAPHICS_STATE_MANAGER

#include "beGraphics.h"
#include <lean/tags/noncopyable.h>
#include <beCore/beShared.h>

namespace beGraphics
{

/// State setup interface.
class StateSetup : public beCore::OptionalResource, public Implementation
{
public:
	virtual ~StateSetup() throw() { }
};

/// State manager interface.
class StateManager : public lean::noncopyable, public beCore::Resource, public Implementation
{
public:
	virtual ~StateManager() throw() { }

	/// Sets the given states.
	virtual void Set(const StateSetup& setup) = 0;
	/// Gets all stored states.
	virtual const StateSetup& Get() const = 0;

	/// Clears the given states.
	virtual void ClearBindings() = 0;
};

} // namespace

#endif