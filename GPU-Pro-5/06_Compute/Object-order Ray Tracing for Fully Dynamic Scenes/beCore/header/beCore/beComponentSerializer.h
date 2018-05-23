/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_SERIALIZER
#define BE_CORE_SERIALIZER

#include "beCore.h"
#include <lean/tags/noncopyable.h>

#include "beComponentInfo.h"

#include <lean/smart/scoped_ptr.h>
#include <lean/rapidxml/rapidxml.hpp>
#include <string>

// Prototypes
namespace beCore
{

class Parameters;
class ParameterSet;
class Component;
struct ComponentType;

class LoadJob;
class SaveJob;
template <class Job>
class SerializationQueue;

/// Serializer.
class LEAN_INTERFACE GenericComponentSerializer : public lean::noncopyable
{
private:
	utf8_string m_type;

public:
	/// Serializable type.
	typedef Component Serializable;

	/// Constructor.
	BE_CORE_API GenericComponentSerializer(const utf8_ntri &type);
	/// Destructor.
	BE_CORE_API virtual ~GenericComponentSerializer();

	/// Sets the name of the serializable object stored in the given xml node.
	BE_CORE_API static void SetName(const utf8_ntri &name, rapidxml::xml_node<lean::utf8_t> &node);
	/// Gets the name of the serializable object stored in the given xml node.
	BE_CORE_API static utf8_ntr GetName(const rapidxml::xml_node<lean::utf8_t> &node);
	/// Gets the type of the serializable object stored in the given xml node.
	BE_CORE_API static utf8_ntr GetType(const rapidxml::xml_node<lean::utf8_t> &node);
	/// Sets the type of the serializable object stored in the given xml node.
	BE_CORE_API static void SetType(const utf8_ntri &type, rapidxml::xml_node<lean::utf8_t> &node);
	/// Gets the ID of the serializable object stored in the given xml node.
	BE_CORE_API static uint8 GetID(const rapidxml::xml_node<lean::utf8_t> &node);
	/// Sets the ID of the serializable object stored in the given xml node.
	BE_CORE_API static void SetID(uint8 id, rapidxml::xml_node<lean::utf8_t> &node);

	/// Gets a list of creation parameters.
	BE_CORE_API virtual ComponentParameters GetCreationParameters() const;
	/// Creates a serializable object from the given parameters.
	virtual Serializable* CreateComponent(const Parameters &creationParameters, const ParameterSet &parameters) const = 0;

	/// Loads a serializable object from the given xml node.
	virtual Serializable* LoadComponent(const rapidxml::xml_node<lean::utf8_t> &node,
		ParameterSet &parameters, SerializationQueue<LoadJob> &queue) const = 0;
	/// Loads a serializable object from the given xml node.
	BE_CORE_API virtual void LoadComponent(Serializable *serializable, const rapidxml::xml_node<lean::utf8_t> &node,
		ParameterSet &parameters, SerializationQueue<LoadJob> &queue) const;
	/// Saves the given serializable object to the given XML node.
	BE_CORE_API virtual void SaveComponent(const Serializable *serializable, rapidxml::xml_node<lean::utf8_t> &node,
		ParameterSet &parameters, SerializationQueue<SaveJob> &queue) const;

	/// Gets the type of objects serialized.
	LEAN_INLINE const utf8_string& GetType() const { return m_type; }
};

/// Serializer.
template <class SerializableType>
class LEAN_INTERFACE ComponentSerializer : public GenericComponentSerializer
{
public:
	/// Serializable type.
	typedef SerializableType Serializable;

private:
	utf8_string m_type;
	
	/// Creates a serializable object from the given parameters.
	GenericComponentSerializer::Serializable* CreateComponent(const Parameters &creationParameters, const ParameterSet &parameters) const LEAN_OVERRIDE
	{
		return Create(creationParameters, parameters).detach();
	}
	/// Loads a serializable object from the given xml node.
	GenericComponentSerializer::Serializable* LoadComponent(const rapidxml::xml_node<lean::utf8_t> &node,
		ParameterSet &parameters, SerializationQueue<LoadJob> &queue) const LEAN_OVERRIDE
	{
		return Load(node, parameters, queue).detach();
	}
	/// Loads a serializable object from the given xml node.
	void LoadComponent(GenericComponentSerializer::Serializable *serializable, const rapidxml::xml_node<lean::utf8_t> &node,
		ParameterSet &parameters, SerializationQueue<LoadJob> &queue) const LEAN_OVERRIDE
	{
		Load(static_cast<Serializable*>(serializable), node, parameters, queue);
	}
	/// Saves the given serializable object to the given XML node.
	void SaveComponent(const GenericComponentSerializer::Serializable *serializable, rapidxml::xml_node<lean::utf8_t> &node,
		ParameterSet &parameters, SerializationQueue<SaveJob> &queue) const LEAN_OVERRIDE
	{
		Save(static_cast<const Serializable*>(serializable), node, parameters, queue);
	}

public:
	/// Constructor.
	LEAN_INLINE ComponentSerializer(const utf8_ntri &type)
		: GenericComponentSerializer(type) { }

	/// Creates a serializable object from the given parameters.
	virtual lean::scoped_ptr<Serializable, lean::critical_ref> Create(const Parameters &creationParameters, const ParameterSet &parameters) const = 0;

	/// Loads a serializable object from the given xml node.
	virtual lean::scoped_ptr<Serializable, lean::critical_ref> Load(const rapidxml::xml_node<lean::utf8_t> &node,
		ParameterSet &parameters, SerializationQueue<LoadJob> &queue) const = 0;
	/// Loads a serializable object from the given xml node.
	virtual void Load(Serializable *serializable, const rapidxml::xml_node<lean::utf8_t> &node,
		ParameterSet &parameters, SerializationQueue<LoadJob> &queue) const
	{
		GenericComponentSerializer::LoadComponent(serializable, node, parameters, queue);
	}
	/// Saves the given serializable object to the given XML node.
	virtual void Save(const Serializable *serializable, rapidxml::xml_node<lean::utf8_t> &node,
		ParameterSet &parameters, SerializationQueue<SaveJob> &queue) const
	{
		GenericComponentSerializer::SaveComponent(serializable, node, parameters, queue);
	}
};

} // namespace

#endif