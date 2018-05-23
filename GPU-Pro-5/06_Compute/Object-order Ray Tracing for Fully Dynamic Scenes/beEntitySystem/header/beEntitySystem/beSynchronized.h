/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_SYNCHRONIZED
#define BE_ENTITYSYSTEM_SYNCHRONIZED

#include "beEntitySystem.h"

namespace beEntitySystem
{

/// Synchronized flag enumeration.
struct SynchronizedFlags
{
	/// Enumeration
	enum T
	{
		None = 0,			///< No snychronized.
		
		Flush = 1 << 0,		///< Requesting flush to be called.
		Fetch = 1 << 1,		///< Requesting fetech to be called.
		
		All = Flush | Fetch	///< Any synchronization.
	};
	LEAN_MAKE_ENUM_STRUCT(SynchronizedFlags)
};

/// Synchronized interface.
class LEAN_INTERFACE Synchronized
{
public:
	/// Synchronizes this synchronized object with a given object.
	virtual void Flush() { }
	/// Synchronizes a given object with this synchronized object.
	virtual void Fetch() { }
};

/// Synchronized interface.
template <class ControllerDriven>
class LEAN_INTERFACE SynchronizedController
{
public:
	/// Synchronizes this synchronized controller with the given controller-driven object.
	virtual void Flush(const ControllerDriven &controllerDriven) { }
	/// Synchronizes the given controller-driven object with this synchronized controller.
	virtual void Fetch(ControllerDriven &controllerDriven) { }
};

} // namespace

#endif