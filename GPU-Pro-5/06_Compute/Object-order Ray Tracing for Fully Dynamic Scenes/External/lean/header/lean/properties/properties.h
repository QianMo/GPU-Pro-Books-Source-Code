/*****************************************************/
/* lean Properties              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_PROPERTIES_PROPERTIES
#define LEAN_PROPERTIES_PROPERTIES

namespace lean
{
	/// Defines classes that allow for the specification of enhanced type information, e.g. making named setter and getter
	/// methods enumerable at run-time, to be used in a generic way while at the same time retaining type safety.
	/// @see PropertyMacros
	namespace properties { }
}

#include "property.h"
#include "property_accessors.h"
#include "property_types.h"
#include "property_collection.h"

#endif