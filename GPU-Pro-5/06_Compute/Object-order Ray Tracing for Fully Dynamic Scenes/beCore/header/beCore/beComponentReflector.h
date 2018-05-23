/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_COMPONENT_REFLECTOR
#define BE_CORE_COMPONENT_REFLECTOR

#include "beCore.h"
#include <lean/tags/noncopyable.h>

#include "beComponent.h"
#include "beComponentInfo.h"

#include "beParameters.h"
#include "beParameterSet.h"

#include <lean/containers/any.h>
#include <lean/smart/cloneable_obj.h>

#include "beExchangeContainers.h"

namespace beCore
{

/// Component state enumeration.
struct ComponentState
{
	// Enum.
	enum T
	{
		NotSet,		///< Null.
		Unknown,	///< Valid, but unmanaged.
		Named,		///< Valid & named.
		Filed		///< Valid & filed.
	};
	LEAN_MAKE_ENUM_STRUCT(ComponentState)
};

/// Component flags.
struct ComponentFlags
{
	/// Enumeration.
	enum T
	{
		None = 0x0,
		
		NameMutable = 0x1,	///< Component name may be changed.
		Filed = 0x2,		///< Component is currently associated with a file.
		FileMutable = 0x4,	///< Component may be associated with any given file.

		Creatable = 0x10,	///< New components may be created.
		Cloneable = 0x20,	///< Existing component may be cloned.

	};
	LEAN_MAKE_ENUM_STRUCT(ComponentFlags)
};

/// Provides generic access to abstract component types.
class LEAN_INTERFACE ComponentReflector
{
	LEAN_INTERFACE_BEHAVIOR(ComponentReflector)

public:
	/// Gets principal component flags.
	virtual uint4 GetComponentFlags() const = 0;
	/// Gets specific component flags.
	virtual uint4 GetComponentFlags(const lean::any &component) const = 0;

	/// Gets information on the components currently available.
	virtual ComponentInfoVector GetComponentInfo(const ParameterSet &parameters) const = 0;
	
	/// Gets the component name.
	virtual ComponentInfo GetInfo(const lean::any &component) const = 0;

	/// Gets a list of creation parameters.
	BE_CORE_API virtual ComponentParameters GetCreationParameters() const { return ComponentParameters(); }
	/// Creates a component from the given parameters.
	BE_CORE_API virtual lean::cloneable_obj<lean::any, true> CreateComponent(const utf8_ntri &name,
		const Parameters &creationParameters, const ParameterSet &parameters,
		const lean::any *pPrototype = nullptr, const lean::any *pReplace = nullptr) const { return nullptr; }
	/// Gets a list of creation parameters.
	BE_CORE_API virtual void GetCreationInfo(const lean::any &component, Parameters &creationParameters, ComponentInfo *pInfo = nullptr) const
	{
		if (pInfo)
			*pInfo = GetInfo(component);
	}

	/// Sets the component name.
	BE_CORE_API virtual void SetName(const lean::any &component, const utf8_ntri &name) const { }
	/// Gets a component by name.
	BE_CORE_API virtual lean::cloneable_obj<lean::any, true> GetComponentByName(const utf8_ntri &name, const ParameterSet &parameters) const { return nullptr; }
	
	/// Sets the component name.
	BE_CORE_API virtual void SetFile(const lean::any &component, const utf8_ntri &file) const { }
	/// Gets a fitting file extension, if available.
	BE_CORE_API virtual utf8_ntr GetFileExtension() const { return utf8_ntr(""); }

	/// Gets a list of loading parameters.
	BE_CORE_API virtual ComponentParameters GetFileParameters(const utf8_ntri &file) const { return ComponentParameters(); }
	/// Gets a component by file.
	BE_CORE_API virtual lean::cloneable_obj<lean::any, true> GetComponentByFile(const utf8_ntri &file,
		const Parameters &fileParameters, const ParameterSet &parameters) const { return nullptr; }
	/// Gets a list of creation parameters.
	BE_CORE_API virtual void GetFileInfo(const lean::any &component, Parameters &fileParameters, ComponentInfo *pInfo = nullptr) const
	{
		if (pInfo)
			*pInfo = GetInfo(component);
	}

	/// Gets the component type reflected.
	virtual const ComponentType* GetType() const = 0;
};

} // namespace

#endif