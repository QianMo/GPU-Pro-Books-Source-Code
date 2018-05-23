/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_PARAMETERS
#define BE_CORE_PARAMETERS

#include "beCore.h"
#include <lean/containers/any.h>
#include <lean/smart/cloneable_obj.h>
#include <lean/meta/strip.h>
#include <vector>

namespace beCore
{

/// Serialization manager.
class Parameters
{
private:
	/// Parameter
	struct Parameter
	{
		utf8_string name;							///< Parameter name.
		lean::cloneable_obj<lean::any, true> value;	///< Parameter value.

		/// Constructor.
		Parameter()
			: value(nullptr) { }
		/// Constructor.
		explicit Parameter(const utf8_ntri &name, lean::any *value = nullptr)
			: name(name.to<utf8_string>()),
			value(value) { }
#ifdef LEAN0X_NEED_EXPLICIT_MOVE
		/// Move constructor.
		Parameter(Parameter &&right)
			: name(std::move(right.name)),
			value(std::move(right.value)) { }
#endif
	};

	typedef std::vector<Parameter> parameter_vector;
	parameter_vector m_parameters;

public:
	/// Invalid parameter ID.
	static const uint4 InvalidID = static_cast<uint4>(-1);

	/// Constructor.
	BE_CORE_API Parameters();
	/// Constructor.
	BE_CORE_API Parameters(const Parameters &right);
	/// Destructor.
	BE_CORE_API ~Parameters();

	/// Adds a parameter of the given name, returning its parameter ID.
	BE_CORE_API uint4 Add(const utf8_ntri &name);
	/// Gets the name of the parameter identified by the given ID.
	BE_CORE_API utf8_ntr GetName(uint4 parameterID) const;
	/// Gets the current number of parameters.
	BE_CORE_API uint4 GetCount() const;
	/// Gets the parameter identified by the given name.
	BE_CORE_API uint4 GetID(const utf8_ntri &name) const;
	
	/// Assigns the given value to the parameter identified by the given ID.
	BE_CORE_API void SetAnyValue(uint4 parameterID, const lean::any *pValue);
	/// Assigns the given value to the parameter identified by the given ID.
	LEAN_INLINE void SetAnyValue(uint4 parameterID, const lean::any &value)
	{
		SetAnyValue(parameterID, &value);
	}
	/// Gets the value of the parameter identified by the given ID.
	BE_CORE_API const lean::any* GetAnyValue(uint4 parameterID) const;

	/// Assigns the given value to the parameter identified by the given ID.
	template <class Value>
	LEAN_INLINE void SetValue(uint4 parameterID, const Value &value)
	{
		lean::any_value<Value> anyValue(value);
		SetAnyValue(parameterID, &anyValue);
	}
	/// Gets the value of the parameter identified by the given ID.
	template <class Value>
	LEAN_INLINE const Value* GetValue(uint4 parameterID) const
	{
		return lean::any_cast<Value>( GetAnyValue(parameterID) );
	}
	/// Gets the value of the parameter identified by the given ID.
	template <class Value>
	LEAN_INLINE const Value& GetValueChecked(uint4 parameterID) const
	{
		return lean::any_cast_checked<const Value&>( GetAnyValue(parameterID) );
	}
	/// Gets the value of the parameter identified by the given ID.
	template <class Value>
	LEAN_INLINE bool GetValue(uint4 parameterID, Value &value) const
	{
		const Value *pValue = GetValue<Value>(parameterID);
		
		if (pValue)
		{
			value = *pValue;
			return true;
		}
		else
			return false;
	}
	/// Gets the value of the parameter identified by the given ID.
	template <class Value>
	LEAN_INLINE Value GetValueDefault(uint4 parameterID, const Value &defaultValue = Value()) const
	{
		Value value;

		if (!GetValue(parameterID, value))
			value = defaultValue;

		return value;
	}

	/// Assigns the given value to the parameter identified by the given name.
	template <class Value>
	LEAN_INLINE void SetValue(const utf8_ntri &name, const Value &value)
	{
		SetValue( Add(name), value );
	}
	/// Gets the value of the parameter identified by the given name.
	template <class Value>
	LEAN_INLINE const Value* GetValue(const utf8_ntri &name) const
	{
		return GetValue<Value>( GetID(name) );
	}
	/// Gets the value of the parameter identified by the given name.
	template <class Value>
	LEAN_INLINE const Value& GetValueChecked(const utf8_ntri &name) const
	{
		return GetValueChecked<Value>( GetID(name) );
	}
	/// Gets the value of the parameter identified by the given name.
	template <class Value>
	LEAN_INLINE bool GetValue(const utf8_ntri &name, Value &value) const
	{
		return GetValue( GetID(name), value );
	}
	/// Gets the value of the parameter identified by the given ID.
	template <class Value>
	LEAN_INLINE Value GetValueDefault(const utf8_ntri &name, const Value &defaultValue = Value()) const
	{
		return GetValueDefault( GetID(name), defaultValue );
	}
};

} // namespace

#endif