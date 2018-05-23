/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beSerializationParameters.h"

namespace beEntitySystem
{

// Gets the serialization parameter layout.
beCore::ParameterLayout& GetSerializationParameters()
{
	static beCore::ParameterLayout manager;
	return manager;
}

// Gets the serialization parameter IDs.
const EntitySystemParameterIDs& GetEntitySystemParameterIDs()
{
	beCore::ParameterLayout &layout = GetSerializationParameters();

	static EntitySystemParameterIDs parameterIDs(
			layout.Add("beEntitySystem.World"),
			layout.Add("beEntitySystem.Entity"),
			layout.Add("beEntitySystem.NoOverwrite")
		);

	return parameterIDs;
}

// Sets the given entity system parameters in the given parameter set.
void SetEntitySystemParameters(beCore::ParameterSet &parameters, const EntitySystemParameters &entitySystemParameters)
{
	const beCore::ParameterLayout &layout = GetSerializationParameters();
	const EntitySystemParameterIDs& parameterIDs = GetEntitySystemParameterIDs();

	parameters.SetValue(layout, parameterIDs.World, entitySystemParameters.World);
	parameters.SetValue(layout, parameterIDs.Entity, entitySystemParameters.Entity);
}

// Sets the given entity system parameters in the given parameter set.
EntitySystemParameters GetEntitySystemParameters(const beCore::ParameterSet &parameters)
{
	EntitySystemParameters entitySystemParameters;

	const beCore::ParameterLayout &layout = GetSerializationParameters();
	const EntitySystemParameterIDs& parameterIDs = GetEntitySystemParameterIDs();

	entitySystemParameters.World = parameters.GetValueChecked< World* >(layout, parameterIDs.World);
	entitySystemParameters.Entity = parameters.GetValueChecked< Entity* >(layout, parameterIDs.Entity);

	return entitySystemParameters;
}

// Sets the given entity system parameters in the given parameter set.
void SetEntityParameter(beCore::ParameterSet &parameters, Entity *pEntity)
{
	const beCore::ParameterLayout &layout = GetSerializationParameters();
	const EntitySystemParameterIDs& parameterIDs = GetEntitySystemParameterIDs();

	parameters.SetValue(layout, parameterIDs.Entity, pEntity);
}

/// Gets the given entity system parameters in the given parameter set.
Entity* GetEntityParameter(const beCore::ParameterSet &parameters)
{
	const beCore::ParameterLayout &layout = GetSerializationParameters();
	const EntitySystemParameterIDs& parameterIDs = GetEntitySystemParameterIDs();

	return parameters.GetValueChecked< Entity* >(layout, parameterIDs.Entity);
}

// Sets the given entity system parameters in the given parameter set.
void SetNoOverwriteParameter(beCore::ParameterSet &parameters, bool bNoOverwrite)
{
	const beCore::ParameterLayout &layout = GetSerializationParameters();
	const EntitySystemParameterIDs& parameterIDs = GetEntitySystemParameterIDs();

	parameters.SetValue(layout, parameterIDs.NoOverwrite, bNoOverwrite);
}

// Gets the given entity system parameters in the given parameter set.
bool GetNoOverwriteParameter(const beCore::ParameterSet &parameters)
{
	const beCore::ParameterLayout &layout = GetSerializationParameters();
	const EntitySystemParameterIDs& parameterIDs = GetEntitySystemParameterIDs();

	return parameters.GetValueDefault< bool >(layout, parameterIDs.NoOverwrite, false);
}


} // namespace
