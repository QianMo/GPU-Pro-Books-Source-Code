/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beSimulation.h"

namespace beEntitySystem
{

// Constructor.
Simulation::Simulation(const utf8_ntri &name)
	: m_name(name.to<utf8_string>()),
	m_bPaused(false)
{
}

// Destructor.
Simulation::~Simulation()
{
}

// Sets the name.
void Simulation::SetName(const utf8_ntri &name)
{
	m_name.assign(name.begin(), name.end());
}

} // namespace
