/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_SERIALIZATION
#define BE_CORE_SERIALIZATION

#include "beCore.h"
#include <lean/tags/noncopyable.h>
#include <lean/smart/scoped_ptr.h>
#include <lean/rapidxml/rapidxml.hpp>
#include <unordered_map>
#include <string>

namespace beCore
{
	
class ParameterSet;
class Component;

class GenericComponentSerializer;
template <class Serializable>
class ComponentSerializer;

class LoadJob;
class SaveJob;
template <class Job>
class SerializationQueue;

/// Serialization manager.
class GenericComponentSerialization : public lean::noncopyable
{
public:
	/// Generic serializer type.
	typedef GenericComponentSerializer Serializer;
	/// Generic serializable type.
	typedef Component Serializable;

private:
	typedef std::unordered_map<utf8_string, const Serializer*> serializer_map;
	serializer_map m_serializers;

public:
	/// Constructor.
	BE_CORE_API GenericComponentSerialization();
	/// Destructor.
	BE_CORE_API ~GenericComponentSerialization();

	/// Adds the given serializer to this serialization manager.
	BE_CORE_API void AddSerializer(const Serializer *serializer);
	/// Removes the given serializer from this serialization manager.
	BE_CORE_API bool RemoveSerializer(const Serializer *serializer);

	/// Gets the number of serializers.
	BE_CORE_API uint4 GetSerializerCount() const;
	/// Gets all serializers.
	BE_CORE_API void GetSerializers(const Serializer **serializers) const;

	/// Gets a serializer for the given serializable type, if available, returns nullptr otherwise.
	BE_CORE_API const Serializer* GetSerializer(const utf8_ntri &type) const;

	/// Loads an entity from the given xml node.
	BE_CORE_API Serializable* Load(const rapidxml::xml_node<lean::utf8_t> &node,
		ParameterSet &parameters, SerializationQueue<LoadJob> &queue) const;
	/// Saves the given serializable object to the given XML node.
	BE_CORE_API bool Save(const Serializable *serializable, rapidxml::xml_node<lean::utf8_t> &node,
		ParameterSet &parameters, SerializationQueue<SaveJob> &queue) const;
};

/// Serialization manager.
template < class SerializableType, class CustomSerializer = ComponentSerializer<SerializableType> >
class ComponentSerialization : private GenericComponentSerialization
{
public:
	/// Serializable type.
	typedef SerializableType Serializable;
	/// Compatible serializer type.
	typedef CustomSerializer Serializer;

	/// Adds the given serializer to this serialization manager.
	LEAN_INLINE void AddSerializer(const Serializer *serializer)
	{
		this->GenericComponentSerialization::AddSerializer(serializer);
	}
	/// Removes the given serializer from this serialization manager.
	LEAN_INLINE bool RemoveSerializer(const Serializer *serializer)
	{
		return this->GenericComponentSerialization::RemoveSerializer(serializer);
	}

	/// Gets the number of serializers.
	LEAN_INLINE uint4 GetSerializerCount() const
	{
		return this->GenericComponentSerialization::GetSerializerCount();
	}
	/// Gets all serializers.
	LEAN_INLINE void GetSerializers(const Serializer **serializers) const
	{
		this->GenericComponentSerialization::GetSerializers(reinterpret_cast<const GenericComponentSerialization::Serializer**>(serializers));
	}

	/// Gets a serializer for the given serializable type, if available, returns nullptr otherwise.
	LEAN_INLINE const Serializer* GetSerializer(const utf8_ntri &type) const
	{
		return static_cast<const Serializer*>(this->GenericComponentSerialization::GetSerializer(type));
	}

	/// Loads an entity from the given xml node.
	LEAN_INLINE lean::scoped_ptr<Serializable, lean::critical_ref> Load(const rapidxml::xml_node<lean::utf8_t> &node,
		ParameterSet &parameters, SerializationQueue<LoadJob> &queue) const
	{
		return lean::scoped_ptr<Serializable, lean::critical_ref>(
				static_cast<Serializable*>(this->GenericComponentSerialization::Load(node, parameters, queue))
			);
	}
	/// Saves the given serializable object to the given XML node.
	LEAN_INLINE bool Save(const Serializable *serializable, rapidxml::xml_node<lean::utf8_t> &node,
		ParameterSet &parameters, SerializationQueue<SaveJob> &queue) const
	{
		return this->GenericComponentSerialization::Save(serializable, node, parameters, queue);
	}
};

/// Instantiate this to add a serializer of the given type the given index of serializers.
template <class SerializerType, class SerializationType, SerializationType& (*GetSerialization)()>
struct ComponentSerializationPlugin
{
	/// Serializer.
	SerializerType Serializer;

	/// Adds the serializer.
	ComponentSerializationPlugin()
	{
		GetSerialization().AddSerializer(&Serializer);
	}
	/// Removes the serializer.
	~ComponentSerializationPlugin()
	{
		GetSerialization().RemoveSerializer(&Serializer);
	}
};

} // namespace

#endif