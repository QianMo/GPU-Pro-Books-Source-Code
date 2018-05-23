/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_SPECIAL_REFLECTION_PROPERTIES
#define BE_CORE_SPECIAL_REFLECTION_PROPERTIES

#include "beCore.h"
#include <lean/properties/property.h>
#include <lean/properties/property_accessors.h>

#include <beMath/beMatrix.h>

namespace beCore
{

template <class Setter>
class PropertyEulerZXYToMatrixSetter : public Setter
{
public:
	typedef typename Setter::object_type object_type;
	typedef typename Setter::original_value_type matrix_type;
	typedef typename matrix_type::component_type value_type;

	/// Passes the given number of values of the given type to the given object using the stored setter method, if the value types are matching.
	bool operator ()(object_type &object, const std::type_info &type, const void *values, size_t count)
	{
		if (count < 3)
			lean::properties::impl::property_error_policy::count_mismatch(3, count);
		else if (type != typeid(value_type))
			lean::properties::impl::property_error_policy::type_mismatch<value_type>(type);
		else
		{
			const value_type *angles = static_cast<const value_type*>(values);
			matrix_type orientation = beMath::mat_rot_zxy<3>(
					angles[0] * beMath::Constants::degrees<value_type>::deg2rad,
					angles[1] * beMath::Constants::degrees<value_type>::deg2rad,
					angles[2] * beMath::Constants::degrees<value_type>::deg2rad
				);
			return Setter::operator ()(object, typeid(orientation), &orientation, 1);
		}

		return false;
	}

	PropertyEulerZXYToMatrixSetter* clone() const { return new PropertyEulerZXYToMatrixSetter(*this); }
	void destroy() const { delete this; }
};

template <class Getter>
class PropertyEulerZXYToMatrixGetter : public Getter
{
public:
	typedef typename Getter::object_type object_type;
	typedef typename Getter::original_value_type matrix_type;
	typedef typename matrix_type::component_type value_type;
	
	/// Retrieves the given number of values of the given type from the given object using the stored getter method, if available.
	bool operator ()(const object_type &object, const std::type_info &type, void *values, size_t count) const
	{
		if (type != typeid(value_type))
			lean::properties::impl::property_error_policy::type_mismatch<value_type>(type);
		else
		{
			matrix_type orientation;
			if (!Getter::operator ()(object, typeid(orientation), &orientation, 1))
				return false;
			
			const beMath::vector<value_type, 3> angles = angles_rot_zxy(orientation) * beMath::Constants::degrees<value_type>::rad2deg;
			memcpy(values, angles.c, sizeof(value_type) * lean::min<size_t>(count, 3));
			return true;
		}

		return false;
	}

	PropertyEulerZXYToMatrixGetter* clone() const { return new PropertyEulerZXYToMatrixGetter(*this); }
	void destroy() const { delete this; }
};

template <class Setter>
LEAN_INLINE PropertyEulerZXYToMatrixSetter<Setter> MakeEulerZXYMatrixSetter(const Setter&)
{
	return PropertyEulerZXYToMatrixSetter<Setter>();
}
template <class Getter>
LEAN_INLINE PropertyEulerZXYToMatrixGetter<Getter> MakeEulerZXYMatrixGetter(const Getter&)
{
	return PropertyEulerZXYToMatrixGetter<Getter>();
}

} // namespace

#endif