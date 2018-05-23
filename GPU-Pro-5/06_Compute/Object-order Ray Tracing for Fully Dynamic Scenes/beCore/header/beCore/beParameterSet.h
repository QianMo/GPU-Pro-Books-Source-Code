/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_PARAMETER_SET
#define BE_CORE_PARAMETER_SET

#include "beCore.h"
#include <lean/containers/any.h>
#include <lean/smart/cloneable_obj.h>
#include <lean/meta/strip.h>
#include <vector>
#include <lean/tags/noncopyable.h>

namespace beCore
{

/// Parameter set.
class ParameterLayout : public lean::nonassignable
{
private:
	typedef std::vector<utf8_string> parameter_vector;
	parameter_vector m_parameters;

public:
	/// Invalid parameter ID.
	static const uint4 InvalidID = static_cast<uint4>(-1);

	/// Constructor.
	BE_CORE_API ParameterLayout();
	/// Constructor.
	BE_CORE_API ParameterLayout(const ParameterLayout &right);
	/// Destructor.
	BE_CORE_API ~ParameterLayout();

	/// Adds a parameter of the given name, returning its parameter ID.
	BE_CORE_API uint4 Add(const utf8_ntri &name);
	/// Gets the name of the parameter identified by the given ID.
	BE_CORE_API utf8_ntr GetName(uint4 parameterID) const;
	/// Gets the current number of parameters.
	BE_CORE_API uint4 GetCount() const;
	/// Gets the parameter identified by the given name.
	BE_CORE_API uint4 GetID(const utf8_ntri &name) const;
};

/// Parameter set.
class ParameterSet : public lean::nonassignable
{
private:
	const ParameterLayout *m_pLayout;

	typedef lean::cloneable_obj<lean::any, true> parameter_value;
	typedef std::vector<parameter_value> parameter_vector;
	parameter_vector m_parameters;

public:
	/// Constructor.
	BE_CORE_API ParameterSet(const ParameterLayout *pLayout);
	/// Constructor.
	BE_CORE_API ParameterSet(const ParameterSet &right);
	/// Destructor.
	BE_CORE_API ~ParameterSet();
	
	/// Assigns the given value to the parameter identified by the given ID.
	BE_CORE_API void SetAnyValue(uint4 parameterID, const lean::any *pValue);
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
	LEAN_INLINE Value GetValueDefault(uint4 parameterID, const Value &defaultValue) const
	{
		Value value(defaultValue);
		GetValue(parameterID, value);
		return value;
	}

	/// Gets the value of the parameter identified by the given ID.
	LEAN_INLINE lean::any* GetAnyValue(uint4 parameterID)
	{
		return const_cast<lean::any*>( const_cast<const ParameterSet*>(this)->GetAnyValue(parameterID) );
	}
	/// Gets the value of the parameter identified by the given ID.
	template <class Value>
	LEAN_INLINE Value* GetValue(uint4 parameterID)
	{
		return const_cast<Value*>( const_cast<const ParameterSet*>(this)->GetValue<Value>(parameterID) );
	}
	/// Gets the value of the parameter identified by the given ID.
	template <class Value>
	LEAN_INLINE Value& GetValueChecked(uint4 parameterID)
	{
		return const_cast<Value&>( const_cast<const ParameterSet*>(this)->GetValueChecked<Value>(parameterID) );
	}

	/// Assigns the given value to the parameter identified by the given ID.
	template <class Value>
	void SetValue(const ParameterLayout &layout, uint4 parameterID, const Value &value)
	{
		if (&layout == m_pLayout)
			SetValue( parameterID, value );
		else
			SetValue( layout.GetName(parameterID), value );
	}
	/// Gets the value of the parameter identified by the given ID.
	template <class Value>
	const Value* GetValue(const ParameterLayout &layout, uint4 parameterID) const
	{
		return (&layout == m_pLayout)
			? GetValue<Value>( parameterID )
			: GetValue<Value>( layout.GetName(parameterID) );
	}
	/// Gets the value of the parameter identified by the given ID.
	template <class Value>
	const Value& GetValueChecked(const ParameterLayout &layout, uint4 parameterID) const
	{
		return (&layout == m_pLayout)
			? GetValueChecked<Value>( parameterID )
			: GetValueChecked<Value>( layout.GetName(parameterID) );
	}
	/// Gets the value of the parameter identified by the given ID.
	template <class Value>
	bool GetValue(const ParameterLayout &layout, uint4 parameterID, Value &value) const
	{
		return (&layout == m_pLayout)
			? GetValue( parameterID, value )
			: GetValue( layout.GetName(parameterID), value );
	}
	/// Gets the value of the parameter identified by the given ID.
	template <class Value>
	Value GetValueDefault(const ParameterLayout &layout, uint4 parameterID, const Value &defaultValue = Value()) const
	{
		Value value(defaultValue);
		GetValue(layout, parameterID, value);
		return value;
	}

	/// Assigns the given value to the parameter identified by the given name.
	template <class Value>
	LEAN_INLINE void SetValue(const utf8_ntri &name, const Value &value)
	{
		SetValue( m_pLayout->GetID(name), value );
	}
	/// Gets the value of the parameter identified by the given name.
	template <class Value>
	LEAN_INLINE const Value* GetValue(const utf8_ntri &name) const
	{
		return GetValue<Value>( m_pLayout->GetID(name) );
	}
	/// Gets the value of the parameter identified by the given name.
	template <class Value>
	LEAN_INLINE const Value& GetValueChecked(const utf8_ntri &name) const
	{
		return GetValueChecked<Value>( m_pLayout->GetID(name) );
	}
	/// Gets the value of the parameter identified by the given name.
	template <class Value>
	LEAN_INLINE bool GetValue(const utf8_ntri &name, Value &value) const
	{
		return GetValue( m_pLayout->GetID(name), value );
	}

	/// Gets the parameter layout.
	LEAN_INLINE const ParameterLayout* GetLayout() const { return m_pLayout; }
};

template <class Value, class Factory>
Value MakeAndSet(ParameterSet &parameters, const ParameterLayout &layout, uint4 parameterID, const Factory &factory)
{
	Value result = factory(parameters, layout);
	parameters.SetValue<Value>(layout, parameterID, result);
	return result;
}

template <class Value, class Factory>
Value MakeAndSet(const ParameterSet &parameters, const ParameterLayout &layout, uint4 parameterID, const Factory &factory)
{
	return factory(parameters, layout);
}

/// Gets the value stored under the given id, creates and stores it if missing.
template <class Value, class Parameters, class Factory>
Value GetOrMake(Parameters &parameters, const ParameterLayout &layout, uint4 parameterID, const Factory &factory)
{
	const Value* pValue = parameters.GetValue<Value>(layout, parameterID);
	return (pValue) ? *pValue : MakeAndSet<Value>(parameters, layout, parameterID, factory);
}

/// Gets the value stored under the given name, creates and stores it if missing.
template <class Value, class Token, class Parameters, class Factory>
Value GetOrMake(Parameters &parameters, ParameterLayout &layout, const utf8_ntri &name, const Factory &factory)
{
	static const uint4 parameterID = layout.Add(name);
	return GetOrMake<Value>(parameters, layout, parameterID, factory);
}

/// Gets the value stored under the given name, creates and stores it if missing.
template <class Value, class Token, class Parameters, class Factory>
Value GetOrMake(Parameters &parameters, const Factory &factory)
{
	return GetOrMake<Value, Token>(parameters, factory.ParameterLayout(), factory.ParameterName(), factory);
}

} // namespace

#endif