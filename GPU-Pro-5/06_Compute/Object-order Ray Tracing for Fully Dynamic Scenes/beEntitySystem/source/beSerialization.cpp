/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beSerialization.h"

namespace beEntitySystem
{

// Gets the entity serialization register.
EntitySerialization& GetEntitySerialization()
{
	static EntitySerialization manager;
	return manager;
}

// Gets the controller serialization register.
EntityControllerSerialization& GetEntityControllerSerialization()
{
	static EntityControllerSerialization manager;
	return manager;
}

// Gets the controller serialization register.
WorldControllerSerialization& GetWorldControllerSerialization()
{
	static WorldControllerSerialization manager;
	return manager;
}

} // namespace
