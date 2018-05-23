/*****************************************************/
/* lean Properties              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_PROPERTIES_PROPERTY_ACCESSORS
#define LEAN_PROPERTIES_PROPERTY_ACCESSORS

// Use short file names in logging
#ifndef LEAN_DEFAULT_FILE_MACRO
	#line __LINE__ "property_accessors.h"
#endif

#include <typeinfo>
#include "../lean.h"
#include "../meta/strip.h"
#include "../logging/log.h"
#include "property.h"

namespace lean
{
namespace properties
{

namespace impl
{

struct property_logging_policy
{
	template <class Property>
	static LEAN_NOINLINE void type_mismatch(const std::type_info &value)
	{
		LEAN_LOG_ERROR("Property / value type mismatch: "
			<< typeid(Property).name() << " vs. " << value.name());
	}

	static LEAN_NOINLINE void count_mismatch(size_t propertyCount, size_t valueCount)
	{
		LEAN_LOG_ERROR("Property / value count mismatch: "
			<< propertyCount << " vs. " << valueCount);
	}
};

typedef property_logging_policy property_error_policy;

} // namespace

#pragma region property_constant

/// Holds a constant array of values, returning a matching subset of these on getter access.
template <class Class, class Value>
class property_constant : public property_getter<Class>
{
private:
	const Value *m_constantValues;
	size_t m_count;

public:
	typedef typename strip_modifiers<Value>::type value_type;
	typedef value_type original_value_type;

	property_constant(const Value *constantValues, size_t count)
		: m_constantValues(constantValues),
		m_count(count) { };

	/// Copies the given number of values of the given type from the constant value array held, if available.
	bool operator ()(const Class &object, const std::type_info &type, void *values, size_t count) const
	{
		if (type == typeid(value_type))
		{
			std::copy_n(m_constantValues, min(count, m_count), static_cast<value_type*>(values));
			return true;
		}
		else
		{
			impl::property_error_policy::type_mismatch<value_type>(type);
			return false;
		}
	}

	property_constant* clone() const { return new property_constant(*this); }
	void destroy() const { delete this; }
};

/// Constructs a property getter that provides access to the given number of the given values.
template <class Class, class Value>
LEAN_INLINE property_constant<Class, Value> make_property_constant(const Value *constantValues, size_t count)
{
	return property_constant<Class, Value>(constantValues, count);
}

#pragma endregion

namespace impl
{

template <class UnionValue, class Value>
struct union_helper
{
	LEAN_STATIC_ASSERT_MSG_ALT(
		(sizeof(UnionValue) >= sizeof(Value))
			? (sizeof(UnionValue) % sizeof(Value) == 0)
			: (sizeof(Value) % sizeof(UnionValue) == 0),
		"Size of union type must be positive multiple of value type.",
		Size_of_union_type_must_be_positive_multiple_of_value_type);

	// Divide by zero prevented by if clause
	#pragma	warning(push)
	#pragma warning(disable : 4723)

	template <class Count>
	static LEAN_INLINE Count union_count(Count count)
	{
		if (sizeof(UnionValue) >= sizeof(Value))
			return count / (sizeof(UnionValue) / sizeof(Value));
		else
			return count * (sizeof(Value) / sizeof(UnionValue));
	}

	#pragma warning(pop)
};

} // namespace

#pragma region property_n_accessors

/// Provides write access to arbitrary object data of the given type using a given setter method.
template <class Class, class UnionValue, class Count, class Return,
	Return (Class::*Setter)(const UnionValue*, Count),
	class Value = UnionValue, class BaseClass = Class>
class property_n_setter : public property_setter<BaseClass>
{
public:
	typedef typename strip_modifiers<Value>::type value_type;
	typedef typename strip_modifiers<UnionValue>::type union_type;
	typedef union_type original_value_type;

	/// Passes the given number of values of the given type to the given object using the stored setter method, if the value types are matching.
	bool operator ()(BaseClass &baseObject, const std::type_info &type, const void *values, size_t count)
	{
		typedef impl::union_helper<union_type, value_type> union_helper;

		Class &object = static_cast<Class&>(baseObject);

		if (type == typeid(value_type))
		{
			(object.*Setter)(
					reinterpret_cast<const union_type*>( static_cast<const value_type*>(values) ),
					static_cast<Count>( union_helper::union_count(count) )
				);
			return true;
		}
		else if (type == typeid(union_type))
		{
			(object.*Setter)(
					static_cast<const union_type*>( values ),
					static_cast<Count>( count )
				);
			return true;
		}
		else
		{
			impl::property_error_policy::type_mismatch<value_type>(type);
			return false;
		}
	}

	property_n_setter* clone() const { return new property_n_setter(*this); }
	void destroy() const { delete this; }
};

/// Provides read access to arbitrary object data of the given type using a given getter method.
template <class Class, class UnionValue, class Count, class Return,
	Return (Class::*Getter)(UnionValue*, Count) const,
	class Value = UnionValue, class BaseClass = Class>
class property_n_getter : public property_getter<BaseClass>
{
public:
	typedef typename strip_modifiers<Value>::type value_type;
	typedef typename strip_modifiers<UnionValue>::type union_type;
	typedef union_type original_value_type;
	
	/// Retrieves the given number of values of the given type from the given object using the stored getter method, if available.
	bool operator ()(const BaseClass &baseObject, const std::type_info &type, void *values, size_t count) const
	{
		typedef impl::union_helper<union_type, value_type> union_helper;

		const Class &object = static_cast<const Class&>(baseObject);

		if (type == typeid(value_type))
		{
			(object.*Getter)(
					reinterpret_cast<union_type*>( static_cast<value_type*>(values) ),
					static_cast<Count>( union_helper::union_count(count) )
				);
			return true;
		}
		else if (type == typeid(union_type))
		{
			(object.*Getter)(
					static_cast<union_type*>( values ),
					static_cast<Count>( count )
				);
			return true;
		}
		else
		{
			impl::property_error_policy::type_mismatch<value_type>(type);
			return false;
		}
	}

	property_n_getter* clone() const { return new property_n_getter(*this); }
	void destroy() const { delete this; }
};

/// property_n_* factory class.
template <class Class, class UnionValue, class Count, class Return,
	class Value = UnionValue, class BaseClass = Class>
struct property_n_accessor_binder
{
	/// Creates a property_n_setter from the given setter.
	template <Return (Class::*Setter)(const UnionValue*, Count)>
	LEAN_INLINE property_n_setter<Class, UnionValue, Count, Return, Setter, Value, BaseClass> bind_setter()
	{
		return property_n_setter<Class, UnionValue, Count, Return, Setter, Value, BaseClass>();
	}

	/// Creates a property_n_getter from the given getter.
	template <Return (Class::*Getter)(UnionValue*, Count) const>
	LEAN_INLINE property_n_getter<Class, UnionValue, Count, Return, Getter, Value, BaseClass> bind_getter()
	{
		return property_n_getter<Class, UnionValue, Count, Return, Getter, Value, BaseClass>();
	}

	/// Replaces the value type of this factory.
	template <class NewValue>
	LEAN_INLINE property_n_accessor_binder<Class, UnionValue, Count, Return, NewValue, BaseClass> set_value()
	{
		return property_n_accessor_binder<Class, UnionValue, Count, Return, NewValue, BaseClass>();
	}

	/// Replaces the base type of this factory.
	template <class NewBase>
	LEAN_INLINE property_n_accessor_binder<Class, UnionValue, Count, Return, Value, NewBase> set_base()
	{
		return property_n_accessor_binder<Class, UnionValue, Count, Return, Value, NewBase>();
	}
};

/// Deduces the property_n_* factory class for the given setter.
template <class Class, class UnionValue, class Count, class Return>
LEAN_INLINE property_n_accessor_binder<Class, UnionValue, Count, Return>
	deduce_accessor_binder(Return (Class::*)(const UnionValue*, Count))
{
	return property_n_accessor_binder<Class, UnionValue, Count, Return>();
}

/// Deduces the property_n_* factory class for the given getter.
template <class Class, class UnionValue, class Count, class Return>
LEAN_INLINE property_n_accessor_binder<Class, UnionValue, Count, Return>
	deduce_accessor_binder(Return (Class::*)(UnionValue*, Count) const)
{
	return property_n_accessor_binder<Class, UnionValue, Count, Return>();
}

#pragma endregion

#pragma region property_c_accessors

namespace impl
{

template <class Class, class ValueArg, int ArgCount = 1, class Return = void>
struct property_c_helper;

template <class Class, class ValueArg, class Return>
struct property_c_helper<Class, ValueArg, 1, Return>
{
	typedef Return (Class::*setter_type)(ValueArg);
	typedef Return (Class::*getter_type)(ValueArg) const;
	static const int argument_count = 1;

	template <class Value>
	static LEAN_INLINE void set(Class &object, setter_type setter, const Value *values)
	{
		(object.*setter)(*values);
	}
	template <class Value>
	static LEAN_INLINE void get(const Class &object, getter_type getter, Value *values)
	{
		(object.*getter)(*values);
	}
};

template <class Class, class ValueArg, class Return>
struct property_c_helper<Class, ValueArg, 2, Return>
{
	typedef Return (Class::*setter_type)(ValueArg, ValueArg);
	typedef Return (Class::*getter_type)(ValueArg, ValueArg) const;
	static const int argument_count = 2;

	template <class Value>
	static LEAN_INLINE void set(Class &object, setter_type setter, const Value *values)
	{
		(object.*setter)(*values, values[1]);
	}
	template <class Value>
	static LEAN_INLINE void get(const Class &object, getter_type getter, Value *values)
	{
		(object.*getter)(*values, values[1]);
	}
};

template <class Class, class ValueArg, class Return>
struct property_c_helper<Class, ValueArg, 3, Return>
{
	typedef Return (Class::*setter_type)(ValueArg, ValueArg, ValueArg);
	typedef Return (Class::*getter_type)(ValueArg, ValueArg, ValueArg) const;
	static const int argument_count = 3;

	template <class Value>
	static LEAN_INLINE void set(Class &object, setter_type setter, const Value *values)
	{
		(object.*setter)(*values, values[1], values[2]);
	}
	template <class Value>
	static LEAN_INLINE void get(const Class &object, getter_type getter, Value *values)
	{
		(object.*getter)(*values, values[1], values[2]);
	}
};

template <class Class, class ValueArg, class Return>
struct property_c_helper<Class, ValueArg, 4, Return>
{
	typedef Return (Class::*setter_type)(ValueArg, ValueArg, ValueArg, ValueArg);
	typedef Return (Class::*getter_type)(ValueArg, ValueArg, ValueArg, ValueArg) const;
	static const int argument_count = 4;

	template <class Value>
	static LEAN_INLINE void set(Class &object, setter_type setter, const Value *values)
	{
		(object.*setter)(*values, values[1], values[2], values[3]);
	}
	template <class Value>
	static LEAN_INLINE void get(const Class &object, getter_type getter, Value *values)
	{
		(object.*getter)(*values, values[1], values[2], values[3]);
	}
};

} // namespace

/// Provides write access to arbitrary object data of the given type using a given multi-parameter setter method.
template <class Class, class UnionValueArg, int ArgCount, class Return,
	typename impl::property_c_helper<Class, UnionValueArg, ArgCount, Return>::setter_type Setter,
	class ValueArg = UnionValueArg, class BaseClass = Class>
class property_c_setter : public property_setter<BaseClass>
{
private:
	typedef impl::property_c_helper<Class, UnionValueArg, ArgCount, Return> setter_helper;

public:
	typedef typename strip_modifiers<typename strip_reference<ValueArg>::type>::type value_type;
	typedef typename strip_modifiers<typename strip_reference<UnionValueArg>::type>::type union_type;
	typedef union_type original_value_type;

	/// Passes the given number of values of the given type to the given object using the stored setter method, if the value types are matching.
	bool operator ()(BaseClass &baseObject, const std::type_info &type, const void *values, size_t count)
	{
		typedef impl::union_helper<union_type, value_type> union_helper;
		
		Class &object = static_cast<Class&>(baseObject);

		if (type == typeid(value_type))
		{
			size_t unionCount = union_helper::union_count(count);

			if (unionCount >= setter_helper::argument_count)
			{
				setter_helper::set(
					object, Setter,
					reinterpret_cast<const union_type*>( static_cast<const value_type*>(values) ) );
				return true;
			}
			else
				impl::property_error_policy::count_mismatch(setter_helper::argument_count, unionCount);
		}
		else if (type == typeid(union_type))
		{
			if (count >= setter_helper::argument_count)
			{
				setter_helper::set(
					object, Setter,
					static_cast<const union_type*>(values) );
				return true;
			}
			else
				impl::property_error_policy::count_mismatch(setter_helper::argument_count, count);
		}
		else
			impl::property_error_policy::type_mismatch<value_type>(type);

		return false;
	}

	property_c_setter* clone() const { return new property_c_setter(*this); }
	void destroy() const { delete this; }
};

/// Provides read access to arbitrary object data of the given type using a given multi-parameter getter method.
template <class Class, class UnionValueArg, int ArgCount, class Return,
	typename impl::property_c_helper<Class, UnionValueArg, ArgCount, Return>::getter_type Getter,
	class ValueArg = UnionValueArg, class BaseClass = Class>
class property_c_getter : public property_getter<BaseClass>
{
private:
	typedef impl::property_c_helper<Class, UnionValueArg, ArgCount, Return> getter_helper;

public:
	typedef typename strip_modifiers<typename strip_reference<ValueArg>::type>::type value_type;
	typedef typename strip_modifiers<typename strip_reference<UnionValueArg>::type>::type union_type;
	typedef union_type original_value_type;

	/// Retrieves the given number of values of the given type from the given object using the stored getter method, if available.
	bool operator ()(const BaseClass &object, const std::type_info &type, void *values, size_t count) const
	{
		typedef impl::union_helper<union_type, value_type> union_helper;

		const Class &object = static_cast<const Class&>(baseObject);

		if (type == typeid(value_type))
		{
			size_t unionCount = union_helper::union_count(count);

			if (unionCount >= getter_helper::argument_count)
			{
				getter_helper::get(
					object, Getter,
					reinterpret_cast<union_type*>( static_cast<value_type*>(values) ) );
				return true;
			}
			else
				impl::property_error_policy::count_mismatch(getter_helper::argument_count, unionCount);
		}
		else if (type == typeid(union_type))
		{
			if (count >= getter_helper::argument_count)
			{
				getter_helper::get(
					object, Getter,
					static_cast<union_type*>(values) );
				return true;
			}
			else
				impl::property_error_policy::count_mismatch(getter_helper::argument_count, count);
		}
		else
			impl::property_error_policy::type_mismatch<value_type>(type);

		return false;
	}

	property_c_getter* clone() const { return new property_c_getter(*this); }
	void destroy() const { delete this; }
};

/// property_c_* factory class.
template <class Class, class UnionValueArg, int ArgCount, class Return,
	class ValueArg = UnionValueArg, class BaseClass = Class>
struct property_c_accessor_binder
{
	/// Creates a property_c_setter from the given setter.
	template <typename impl::property_c_helper<Class, UnionValueArg, ArgCount, Return>::setter_type Setter>
	LEAN_INLINE property_c_setter<Class, UnionValueArg, ArgCount, Return, Setter, ValueArg, BaseClass> bind_setter()
	{
		return property_c_setter<Class, UnionValueArg, ArgCount, Return, Setter, ValueArg, BaseClass>();
	}
	
	/// Creates a property_c_getter from the given getter.
	template <typename impl::property_c_helper<Class, UnionValueArg, ArgCount, Return>::getter_type Getter>
	LEAN_INLINE property_c_getter<Class, UnionValueArg, ArgCount, Return, Getter, ValueArg, BaseClass> bind_getter()
	{
		return property_c_getter<Class, UnionValueArg, ArgCount, Return, Getter, ValueArg, BaseClass>();
	}

	/// Replaces the value type of this factory.
	template <class NewValue>
	LEAN_INLINE property_c_accessor_binder<Class, UnionValueArg, ArgCount, Return, NewValue, BaseClass> set_value()
	{
		return property_c_accessor_binder<Class, UnionValueArg, ArgCount, Return, NewValue, BaseClass>();
	}

	/// Replaces the base type of this factory.
	template <class NewBase>
	LEAN_INLINE property_c_accessor_binder<Class, UnionValueArg, ArgCount, Return, ValueArg, NewBase> set_base()
	{
		return property_c_accessor_binder<Class, UnionValueArg, ArgCount, Return, ValueArg, NewBase>();
	}
};

/// Deduces the property_c_* factory class for the given setter.
template <class Class, class UnionValueArg, class Return>
LEAN_INLINE property_c_accessor_binder<Class, UnionValueArg, 1, Return>
	deduce_accessor_binder(Return (Class::*)(UnionValueArg))
{
	return property_c_accessor_binder<Class, UnionValueArg, 1, Return>();
}
/// Deduces the property_c_* factory class for the given getter.
template <class Class, class UnionValueArg, class Return>
LEAN_INLINE property_c_accessor_binder<Class, UnionValueArg, 1, Return>
	deduce_accessor_binder(Return (Class::*)(UnionValueArg) const)
{
	return property_c_accessor_binder<Class, UnionValueArg, 1, Return>();
}

/// Deduces the property_c_* factory class for the given setter.
template <class Class, class UnionValueArg, class Return>
LEAN_INLINE property_c_accessor_binder<Class, UnionValueArg, 2, Return>
	deduce_accessor_binder(Return (Class::*)(UnionValueArg, UnionValueArg))
{
	return property_c_accessor_binder<Class, UnionValueArg, 2, Return>();
}
/// Deduces the property_c_* factory class for the given getter.
template <class Class, class UnionValueArg, class Return>
LEAN_INLINE property_c_accessor_binder<Class, UnionValueArg, 2, Return>
	deduce_accessor_binder(Return (Class::*)(UnionValueArg, UnionValueArg) const)
{
	return property_c_accessor_binder<Class, UnionValueArg, 2, Return>();
}

/// Deduces the property_c_* factory class for the given setter.
template <class Class, class UnionValueArg, class Return>
LEAN_INLINE property_c_accessor_binder<Class, UnionValueArg, 3, Return>
	deduce_accessor_binder(Return (Class::*)(UnionValueArg, UnionValueArg, UnionValueArg))
{
	return property_c_accessor_binder<Class, UnionValueArg, 3, Return>();
}
/// Deduces the property_c_* factory class for the given getter.
template <class Class, class UnionValueArg, class Return>
LEAN_INLINE property_c_accessor_binder<Class, UnionValueArg, 3, Return>
	deduce_accessor_binder(Return (Class::*)(UnionValueArg, UnionValueArg, UnionValueArg) const)
{
	return property_c_accessor_binder<Class, UnionValueArg, 3, Return>();
}

/// Deduces the property_c_* factory class for the given setter.
template <class Class, class UnionValueArg, class Return>
LEAN_INLINE property_c_accessor_binder<Class, UnionValueArg, 4, Return>
	deduce_accessor_binder(Return (Class::*)(UnionValueArg, UnionValueArg, UnionValueArg, UnionValueArg))
{
	return property_c_accessor_binder<Class, UnionValueArg, 4, Return>();
}
/// Deduces the property_c_* factory class for the given getter.
template <class Class, class UnionValueArg, class Return>
LEAN_INLINE property_c_accessor_binder<Class, UnionValueArg, 4, Return>
	deduce_accessor_binder(Return (Class::*)(UnionValueArg, UnionValueArg, UnionValueArg, UnionValueArg) const)
{
	return property_c_accessor_binder<Class, UnionValueArg, 4, Return>();
}

#pragma endregion

#pragma region property_r_accessors

/// Provides read access to arbitrary object data of the given type using a given return-value getter method.
template <class Class, class UnionValueReturn,
	UnionValueReturn (Class::*Getter)() const,
	class ValueReturn = UnionValueReturn, class BaseClass = Class>
class property_r_getter : public property_getter<BaseClass>
{
public:
	typedef typename strip_modifiers<typename strip_reference<ValueReturn>::type>::type value_type;
	typedef typename strip_modifiers<typename strip_reference<UnionValueReturn>::type>::type union_type;
	typedef union_type original_value_type;

	/// Retrieves the given number of values of the given type from the given object using the stored getter method, if available.
	bool operator ()(const BaseClass &baseObject, const std::type_info &type, void *values, size_t count) const
	{
		typedef impl::union_helper<union_type, value_type> union_helper;

		const Class &object = static_cast<const Class&>(baseObject);

		if (type == typeid(value_type))
		{
			size_t unionCount = union_helper::union_count(count);

			if (unionCount >= 1)
			{
				*reinterpret_cast<union_type*>( static_cast<value_type*>(values) ) = (object.*Getter)();
				return true;
			}
			else
				impl::property_error_policy::count_mismatch(1, unionCount);
		}
		else if (type == typeid(union_type))
		{
			if (count >= 1)
			{
				*static_cast<union_type*>(values) = (object.*Getter)();
				return true;
			}
			else
				impl::property_error_policy::count_mismatch(1, count);
		}
		else
			impl::property_error_policy::type_mismatch<value_type>(type);

		return false;
	}

	property_r_getter* clone() const { return new property_r_getter(*this); }
	void destroy() const { delete this; }
};

/// property_r_* factory class.
template <class Class, class UnionValueReturn,
	class ValueReturn = UnionValueReturn, class BaseClass = Class>
struct property_r_accessor_binder
{
	/// Creates a property_r_getter from the given getter.
	template <UnionValueReturn (Class::*Getter)() const>
	LEAN_INLINE property_r_getter<Class, UnionValueReturn, Getter, ValueReturn, BaseClass> bind_getter()
	{
		return property_r_getter<Class, UnionValueReturn, Getter, ValueReturn, BaseClass>();
	}

	/// Replaces the value type of this factory.
	template <class NewValue>
	LEAN_INLINE property_r_accessor_binder<Class, UnionValueReturn, NewValue, BaseClass> set_value()
	{
		return property_r_accessor_binder<Class, UnionValueReturn, NewValue, BaseClass>();
	}

	/// Replaces the base type of this factory.
	template <class NewBase>
	LEAN_INLINE property_r_accessor_binder<Class, UnionValueReturn, ValueReturn, NewBase> set_base()
	{
		return property_r_accessor_binder<Class, UnionValueReturn, ValueReturn, NewBase>();
	}
};

/// Deduces the property_r_* factory class for the given getter.
template <class Class, class UnionValueReturn>
LEAN_INLINE property_r_accessor_binder<Class, UnionValueReturn>
	deduce_accessor_binder(UnionValueReturn (Class::*Getter)() const)
{
	return property_r_accessor_binder<Class, UnionValueReturn>();
}

#pragma endregion

} // namespace

using properties::make_property_constant;

} // namespace

/// @addtogroup PropertyMacros Property macros
/// @see lean::properties
/// @{

/// Constructs a property getter that provides access to the given number of the given values.
#define LEAN_MAKE_PROPERTY_CONSTANT(clazz, constants, count) ::lean::properties::make_property_constant<clazz>(constants, count)

/// Constructs a property setter that provides access to object values using the given setter method.
#define LEAN_MAKE_PROPERTY_SETTER(setter) ::lean::properties::deduce_accessor_binder(setter).bind_setter<setter>()
/// Constructs a property setter that provides access to object values using the given getter method.
#define LEAN_MAKE_PROPERTY_GETTER(getter) ::lean::properties::deduce_accessor_binder(getter).bind_getter<getter>()

/// Constructs a property setter that provides access to object values using the given setter method, splitting or merging values of the given type to values of the setter parameter type.
#define LEAN_MAKE_PROPERTY_SETTER_UNION(setter, value_type) ::lean::properties::deduce_accessor_binder(setter).set_value<value_type>().bind_setter<setter>()
/// Constructs a property getter that provides access to object values using the given getter method, splitting or merging values of the given type to values of the getter parameter (return) type.
#define LEAN_MAKE_PROPERTY_GETTER_UNION(getter, value_type) ::lean::properties::deduce_accessor_binder(getter).set_value<value_type>().bind_getter<getter>()

/// @}

#endif