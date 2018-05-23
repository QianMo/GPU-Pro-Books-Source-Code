/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_SERIALIZATION
#define BE_ENTITYSYSTEM_SERIALIZATION

#include "beEntitySystem.h"
#include <beCore/beComponentSerialization.h>

namespace beEntitySystem
{

class Entity;
class EntitySerializer;

class Controller;
class ControllerSerializer;

class EntityController;
class WorldController;

/// Entity serialization.
typedef beCore::ComponentSerialization<Entity, EntitySerializer> EntitySerialization;
/// Controller serialization.
typedef beCore::ComponentSerialization<EntityController, ControllerSerializer> EntityControllerSerialization;
/// Controller serialization.
typedef beCore::ComponentSerialization<WorldController, ControllerSerializer> WorldControllerSerialization;

/// Gets the entity serialization register.
BE_ENTITYSYSTEM_API EntitySerialization& GetEntitySerialization();
/// Gets the entity controller serialization register.
BE_ENTITYSYSTEM_API EntityControllerSerialization& GetEntityControllerSerialization();
/// Gets the world controller serialization register.
BE_ENTITYSYSTEM_API WorldControllerSerialization& GetWorldControllerSerialization();

/// Instantiate this to add a serializer of the given type.
template <class EntitySerializer>
struct EntitySerializationPlugin : public beCore::ComponentSerializationPlugin<EntitySerializer, EntitySerialization, &GetEntitySerialization> { };
/// Instantiate this to add a serializer of the given type.
template <class ControllerSerializer>
struct EntityControllerSerializationPlugin : public beCore::ComponentSerializationPlugin<ControllerSerializer, EntityControllerSerialization, &GetEntityControllerSerialization> { };
/// Instantiate this to add a serializer of the given type.
template <class ControllerSerializer>
struct WorldControllerSerializationPlugin : public beCore::ComponentSerializationPlugin<ControllerSerializer, WorldControllerSerialization, &GetWorldControllerSerialization> { };

} // namespace

#endif