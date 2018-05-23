/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_AAB_DEF
#define BE_MATH_AAB_DEF

#include "beMath.h"
#include "beAABFwd.h"
#include "beTuple.h"
#include "beVectorDef.h"
#include <lean/limits.h>

namespace beMath
{

/// Axis-aligned box class.
template <class Component, size_t Dimension>
class aab
{
public:
	/// Component type.
	typedef Component component_type;
	/// Element count.
	static const size_t dimension = Dimension;
	/// Position type.
	typedef vector<component_type, dimension> position_type;

	/// (Smallest) invalid box.
	static const aab invalid;

	/// Minimum.
	position_type min;
	/// Maximum.
	position_type max;

	/// Creates a default-initialized axis-aligned box.
	LEAN_INLINE aab()
		: min(),
		max() { }
	/// Creates an uninitialized axis-aligned box.
	LEAN_INLINE aab(uninitialized_t)
		: min(uninitialized),
		max(uninitialized) { }
	/// Initializes an axis-aligned box with the given vectors.
	template <class MinClass, class MaxClass>
	LEAN_INLINE aab(
		const class tuple<MinClass, component_type, dimension> &min,
		const class tuple<MaxClass, component_type, dimension> &max)
			: min(min),
			max(max) { }

	/// Gets a raw data pointer.
	LEAN_INLINE component_type* data() { return m_min.data(); }
	/// Gets a raw data pointer.
	LEAN_INLINE const component_type* data() const { return m_min.data(); }
	/// Gets a raw data pointer.
	LEAN_INLINE const component_type* cdata() const { return m_min.cdata(); }
};

// Identity matrix.
template <class Component, size_t Dimension>
const aab<Component, Dimension> aab<Component, Dimension>::invalid = aab<Component, Dimension>(
		// MONITOR: ONLY works if the compiler treats both constants as constexpr, otherwise unordered template constant initialization strikes!
		vector<Component, Dimension>(lean::numeric_limits<Component>::max),
		vector<Component, Dimension>(lean::numeric_limits<Component>::min)
	);

} // namespace

#endif