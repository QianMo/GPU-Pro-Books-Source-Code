/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_COMPONENT_INFO
#define BE_CORE_COMPONENT_INFO

#include "beCore.h"
#include "beExchangeContainers.h"
#include "beComponent.h"

namespace beCore
{

/// Component information.
struct ComponentInfo
{
	Exchange::utf8_string Name;
	Exchange::utf8_string File;
	Exchange::utf8_string Notes;

	ComponentInfo() { }
	ComponentInfo(utf8_ntri name, utf8_ntri file, utf8_ntri notes)
		: Name(name.to<Exchange::utf8_string>()),
		File(file.to<Exchange::utf8_string>()),
		Notes(notes.to<Exchange::utf8_string>()) { }
};
/// List of component information records.
typedef Exchange::vector_t<ComponentInfo>::t ComponentInfoVector;

/// Component parameter flags
struct ComponentParameterFlags
{
	enum T
	{
		None = 0,			///< Default.
		Optional = 0x1,		///< True, if parameter optional.
		Deducible = 0x2,	///< True, if parameter deducible from a given prototype.
		Array = 0x4			///< True, if parameter is an array.
	};
};

/// Specific parameter.
struct ComponentParameter
{
	utf8_ntr Name;				///< Parameter name.
	const ComponentType *Type;	///< Parameter type.
	uint4 Flags;				///< Parameter / type flags.

	/// Constructor
	ComponentParameter(const utf8_ntr &name, const ComponentType *type, uint4 flags = ComponentParameterFlags::None)
		: Name(name),
		Type(type),
		Flags(flags) { }
};
/// Component parameter range.
typedef lean::range<const ComponentParameter*> ComponentParameters;

} // namespace

#endif