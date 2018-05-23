/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_SIMULATION
#define BE_ENTITYSYSTEM_SIMULATION

#include "beEntitySystem.h"
#include <beCore/beShared.h>
#include <beCore/beComponent.h>
#include "beSynchronizedHost.h"
#include "beAnimatedHost.h"
#include "beRenderableHost.h"
#include <beCore/beExchangeContainers.h>

namespace beEntitySystem
{

/// Simulation class.
class Simulation : public beCore::Resource,
	public SynchronizedHost, public AnimatedHost, public RenderableHost
{
private:
	utf8_string m_name;
	bool m_bPaused;

protected:
	Simulation& operator =(const Simulation&) { return *this; }

public:
	/// Constructor.
	BE_ENTITYSYSTEM_API Simulation(const utf8_ntri &name);
	/// Destructor.
	BE_ENTITYSYSTEM_API virtual ~Simulation();

	/// Pases the simulation.
	LEAN_INLINE void Pause(bool bPause) { m_bPaused = bPause; }
	/// Gets whether the simulation is currently paused.
	LEAN_INLINE bool IsPaused() const { return m_bPaused; }

	/// Sets the name.
	BE_ENTITYSYSTEM_API void SetName(const utf8_ntri &name);
	/// Gets the name.
	LEAN_INLINE const utf8_string& GetName() const { return m_name; }
};

} // namespace

#endif