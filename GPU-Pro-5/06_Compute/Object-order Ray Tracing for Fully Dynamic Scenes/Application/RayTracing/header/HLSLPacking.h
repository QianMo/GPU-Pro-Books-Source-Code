#pragma once

#include "Tracing.h"

#include <beMath/beVectorDef.h>
#include <beMath/beMatrixDef.h>

namespace hlsl
{

using namespace breeze;
using namespace lean::types;
LEAN_REIMPORT_NUMERIC_TYPES;

namespace pad
{

template <class T, size_t X>
struct padded_vector_base
{
	LEAN_STATIC_ASSERT(X <= 4);

	bem::vector<T, X> v;
	T _[4 - X];
};

template <class T>
struct padded_vector_base<T, 4>
{
	bem::vector<T, 4> v;
};

template <class T, size_t X>
struct vector : public padded_vector_base<T, X>
{
	typedef bem::vector<T, X> value_type;

	LEAN_INLINE vector() { }
	LEAN_INLINE vector(const value_type &v) { this->v = v; }

	LEAN_INLINE vector& operator =(const value_type &v) { this->v = v; return *this; }

	LEAN_INLINE typename value_type::component_type& operator [](size_t n) { return this->v[n]; }
	LEAN_INLINE const typename value_type::component_type& operator [](size_t n) const { return this->v[n]; }

	LEAN_INLINE typename value_type::component_type* data() { return this->v.data(); }
	LEAN_INLINE const typename value_type::component_type* data() const { return this->v.data(); }
	LEAN_INLINE const typename value_type::component_type* cdata() const { return this->v.cdata(); }
};

template <class T, size_t X, size_t Y>
struct matrix
{
	typedef bem::matrix<T, X, Y> value_type;

	vector<T, Y> r[X];

	LEAN_INLINE matrix() { }
	LEAN_INLINE matrix(const value_type &m)
	{
		*this = m;
	}

	LEAN_INLINE matrix& operator =(const value_type &m)
	{
		for (size_t i = 0; i < X; ++i)
			this->r[i] = m[i];
		return *this;
	}

	LEAN_INLINE typename value_type::row_type& operator [](size_t n) { return this->r[n].v; }
	LEAN_INLINE const typename value_type::row_type& operator [](size_t n) const { return this->r[n].v; }

	LEAN_INLINE typename value_type::component_type* data() { return this->r[0].data(); }
	LEAN_INLINE const typename value_type::component_type* data() const { return this->r[0].data(); }
	LEAN_INLINE const typename value_type::component_type* cdata() const { return this->r[0].cdata(); }
};

typedef matrix<lean::uint4, 2, 2> uint2x2;
typedef matrix<lean::uint4, 3, 3> uint3x3;
typedef matrix<lean::uint4, 4, 4> uint4x4;

typedef matrix<lean::int4, 2, 2> int2x2;
typedef matrix<lean::int4, 3, 3> int3x3;
typedef matrix<lean::int4, 4, 4> int4x4;

typedef matrix<lean::float4, 2, 2> float2x2;
typedef matrix<lean::float4, 3, 3> float3x3;
typedef matrix<lean::float4, 4, 4> float4x4;

} // namespace

} // namespace
