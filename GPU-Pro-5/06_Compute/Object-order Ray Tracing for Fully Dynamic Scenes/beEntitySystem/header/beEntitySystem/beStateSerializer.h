/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_STATESERIALIZER
#define BE_ENTITYSYSTEM_STATESERIALIZER

#include "beEntitySystem.h"

namespace beEntitySystem
{

/// State serializer.
template <class Stateful>
class StateSerializer : public lean::noncopyable
{
private:
	utf8_string m_type;

public:
	/// Stateful type.
	typedef Stateful Stateful;

	/// Constructor.
	BE_ENTITYSYSTEM_API StateSerializer(const utf8_ntri &type);
	/// Destructor.
	BE_ENTITYSYSTEM_API ~StateSerializer();

	/// Restores the state of the given stateful object from the given memento.
	BE_ENTITYSYSTEM_API virtual void Restore(Stateful *pStateful, const rapidxml::xml_node<lean::utf8_t> &node, const beCore::Parameters &parameters) const;
	/// Saves the state of the given stateful object to a memento.
	BE_ENTITYSYSTEM_API virtual void Save(const Stateful *pStateful, rapidxml::xml_node<lean::utf8_t> &node) const;

	/// Gets the type of objects serialized.
	LEAN_INLINE const utf8_string& GetType() const { return m_type; }
};

}

#endif