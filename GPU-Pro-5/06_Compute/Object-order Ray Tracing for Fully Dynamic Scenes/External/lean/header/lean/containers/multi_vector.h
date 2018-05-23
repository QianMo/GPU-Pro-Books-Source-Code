/*****************************************************/
/* lean Containers              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONTAINERS_MULTI_VECTOR
#define LEAN_CONTAINERS_MULTI_VECTOR

#include "../lean.h"
#include "../functional/variadic.h"
#include "vector_policies.h"

namespace lean 
{
namespace containers
{
	
struct multi_vector_base
{
	LEAN_INLINE multi_vector_base() { }
	template <class T1>
	LEAN_INLINE multi_vector_base(T1 LEAN_FW_REF) { }
	template <class T1, class T2>
	LEAN_INLINE multi_vector_base(T1 LEAN_FW_REF, T2 LEAN_FW_REF) { }

	LEAN_INLINE void operator ()(struct ignore&) const { }
	LEAN_INLINE void get(struct ignore&) const { }
	LEAN_INLINE void push_back() { }
	LEAN_INLINE void push_back_from(const multi_vector_base&, size_t) { }
	LEAN_INLINE void erase(size_t idx) { }
	LEAN_INLINE void clear() { }
	LEAN_INLINE void resize(size_t size) { }
	LEAN_INLINE void reserve(size_t size) { }
	LEAN_INLINE void swap(multi_vector_base&) { }
};

/// Manages several parallel vectors.
template <class Type, class Tag, class VectorBinder, class Base = multi_vector_base>
class multi_vector : private Base
{
public:
	typedef typename VectorBinder::template rebind<Type>::type vector_type;
	typedef typename vector_type::allocator_type allocator_type;

private:
	vector_type v;

	LEAN_NOINLINE void pop_or_terminate()
	{
		try
		{
			v.pop_back();
		}
		catch (...)
		{
			LEAN_ASSERT_DEBUG(false);
			std::terminate();
		}
	}

	LEAN_NOINLINE void restore_or_terminate(size_t oldSize)
	{
		try
		{
			v.resize(oldSize);
		}
		catch (...)
		{
			LEAN_ASSERT_DEBUG(false);
			std::terminate();
		}
	}

public:
	LEAN_INLINE multi_vector() { }
	LEAN_INLINE multi_vector(const allocator_type &allocator)
		: Base(allocator),
		v(allocator) { }
	LEAN_INLINE multi_vector(size_t size, const allocator_type &allocator = allocator_type())
		: Base(size, allocator),
		v(size, allocator) { }

#ifdef LEAN0X_NEED_EXPLICIT_MOVE
	LEAN_INLINE multi_vector(multi_vector &&right)
		: Base(std::move(right)),
		v(std::move(right.v)) { }

	LEAN_INLINE multi_vector& operator =(multi_vector &&right)
	{
		this->Base::operator =(std::move(right));
		v = std::move(right.v);
		return *this;
	}
#endif

	using Base::get;
	LEAN_INLINE vector_type& get(Tag) { return v; }
	LEAN_INLINE const vector_type& get(Tag) const { return v; }

	using Base::operator ();
	LEAN_INLINE vector_type& operator ()(Tag) { return v; }
	LEAN_INLINE const vector_type& operator ()(Tag) const { return v; }
	
	void push_back()
	{
		v.push_back(Type());

		try
		{
			this->Base::push_back();
		}
		catch (...)
		{
			pop_or_terminate();
			throw;
		}
	}
	
	#define LEAN_MULTI_VECTOR_PUSH_BACK_METHOD_TPARAMS \
		class T
	#define LEAN_MULTI_VECTOR_PUSH_BACK_METHOD_DECL \
		void push_back
	#define LEAN_MULTI_VECTOR_PUSH_BACK_METHOD_PARAMS \
		T LEAN_FW_REF val
	#define LEAN_MULTI_VECTOR_PUSH_BACK_METHOD_BODY(call) \
		{ \
			v.push_back( LEAN_FORWARD(T, val) ); \
			try \
			{ \
				this->Base::push_back##call; \
			} \
			catch (...) \
			{ \
				pop_or_terminate(); \
				throw; \
			} \
		}
	LEAN_VARIADIC_TEMPLATE_TP(LEAN_FORWARD, LEAN_MULTI_VECTOR_PUSH_BACK_METHOD_DECL, LEAN_MULTI_VECTOR_PUSH_BACK_METHOD_TPARAMS,
		LEAN_MULTI_VECTOR_PUSH_BACK_METHOD_PARAMS, LEAN_NOTHING, LEAN_MULTI_VECTOR_PUSH_BACK_METHOD_BODY)

	LEAN_INLINE void push_back_from(const multi_vector& src, size_t index)
	{
		v.push_back(src.v[index]);

		try
		{
			this->Base::push_back_from(src, index);
		}
		catch (...)
		{
			pop_or_terminate();
			throw;
		}
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	LEAN_INLINE void push_back_from(multi_vector &&src, size_t index)
	{
		v.push_back(LEAN_MOVE(src.v[index]));

		try
		{
			this->Base::push_back_from(LEAN_MOVE(src), index);
		}
		catch (...)
		{
			pop_or_terminate();
			throw;
		}
	}
#endif

	void erase(size_t idx)
	{
		v.erase(v.begin() + idx);
		try
		{
			this->Base::erase(idx);
		}
		catch (...)
		{
			LEAN_ASSERT_DEBUG(false);
			std::terminate();
		}
	}
	
	void clear()
	{
		v.clear();
		try
		{
			this->Base::clear();
		}
		catch (...)
		{
			LEAN_ASSERT_DEBUG(false);
			std::terminate();
		}
	}

	void resize(size_t size)
	{
		size_t oldSize = v.size();
		v.resize(size);
		try
		{
			this->Base::resize(size);
		}
		catch (...)
		{
			restore_or_terminate(oldSize);
			throw;
		}
	}
	LEAN_INLINE void reserve(size_t size) { v.reserve(size); this->Base::reserve(size); }

	LEAN_INLINE size_t size() const { return v.size(); }

	LEAN_INLINE Type& operator [](size_t idx) { return v[idx]; }
	LEAN_INLINE const Type& operator [](size_t idx) const { return v[idx]; }

	LEAN_INLINE vector_type& current() { return v; }
	LEAN_INLINE const vector_type& current() const { return v; }
	LEAN_INLINE Base& next() { return *this; }
	LEAN_INLINE const Base& next() const { return *this; }

	LEAN_INLINE void swap(multi_vector &right)
	{
		using std::swap;
		swap(v, right.v);
		this->Base::swap(right);
	}
};

/// Swaps the given two multi_vectors.
template <class Type, class ID, class VectorBinder, class Base>
LEAN_INLINE void swap(multi_vector<Type, ID, VectorBinder, Base> &left, multi_vector<Type, ID, VectorBinder, Base> &right)
{
	left.swap(right);
}

/// Swizzles the given multi-vector using the given index array.
template <class Type, class ID, class VectorBinder, class Base, class Source, class Iterator>
LEAN_INLINE void append_swizzled(Source LEAN_FW_REF src, Iterator begin, Iterator end, multi_vector<Type, ID, VectorBinder, Base> &dest)
{
	dest.reserve(dest.size() + (end - begin));
	while (begin < end)
		dest.push_back_from(LEAN_FORWARD(Source, src), *begin++);
}

template <class VectorBinder>
struct multi_vector_t
{
	template <class Type1, class ID1,
		class Type2 = void, class ID2 = void,
		class Type3 = void, class ID3 = void,
		class Type4 = void, class ID4 = void,
		class Type5 = void, class ID5 = void,
		class Type6 = void, class ID6 = void,
		class Type7 = void, class ID7 = void,
		class Type8 = void, class ID8 = void,
		class Type9 = void, class ID9 = void>
	struct make
	{
		typedef multi_vector<Type1, ID1, VectorBinder, typename multi_vector_t::template make<Type2, ID2, Type3, ID3, Type4, ID4, Type5, ID5, Type6, ID6, Type7, ID7, Type8, ID8, Type9, ID9>::type> type;
	};

	template <class Type1, class ID1>
	struct make<Type1, ID1>
	{
		typedef multi_vector<Type1, ID1, VectorBinder> type;
	};
	template <class Type1, class ID1, class Type2, class ID2>
	struct make<Type1, ID1, Type2, ID2>
	{
		typedef multi_vector<Type1, ID1, VectorBinder, typename multi_vector_t::template make<Type2, ID2>::type> type;
	};
	template <class Type1, class ID1, class Type2, class ID2, class Type3, class ID3>
	struct make<Type1, ID1, Type2, ID2, Type3, ID3>
	{
		typedef multi_vector<Type1, ID1, VectorBinder, typename multi_vector_t::template make<Type2, ID2, Type3, ID3>::type> type;
	};
	template <class Type1, class ID1, class Type2, class ID2, class Type3, class ID3, class Type4, class ID4>
	struct make<Type1, ID1, Type2, ID2, Type3, ID3, Type4, ID4>
	{
		typedef multi_vector<Type1, ID1, VectorBinder, typename multi_vector_t::template make<Type2, ID2, Type3, ID3, Type4, ID4>::type> type;
	};
	template <class Type1, class ID1, class Type2, class ID2, class Type3, class ID3, class Type4, class ID4, class Type5, class ID5>
	struct make<Type1, ID1, Type2, ID2, Type3, ID3, Type4, ID4, Type5, ID5>
	{
		typedef multi_vector<Type1, ID1, VectorBinder, typename multi_vector_t::template make<Type2, ID2, Type3, ID3, Type4, ID4, Type5, ID5>::type> type;
	};
	template <class Type1, class ID1, class Type2, class ID2, class Type3, class ID3, class Type4, class ID4, class Type5, class ID5, class Type6, class ID6>
	struct make<Type1, ID1, Type2, ID2, Type3, ID3, Type4, ID4, Type5, ID5, Type6, ID6>
	{
		typedef multi_vector<Type1, ID1, VectorBinder, typename multi_vector_t::template make<Type2, ID2, Type3, ID3, Type4, ID4, Type5, ID5, Type6, ID6>::type> type;
	};
	template <class Type1, class ID1, class Type2, class ID2, class Type3, class ID3, class Type4, class ID4, class Type5, class ID5, class Type6, class ID6, class Type7, class ID7>
	struct make<Type1, ID1, Type2, ID2, Type3, ID3, Type4, ID4, Type5, ID5, Type6, ID6, Type7, ID7>
	{
		typedef multi_vector<Type1, ID1, VectorBinder, typename multi_vector_t::template make<Type2, ID2, Type3, ID3, Type4, ID4, Type5, ID5, Type6, ID6, Type7, ID7>::type> type;
	};
	template <class Type1, class ID1, class Type2, class ID2, class Type3, class ID3, class Type4, class ID4, class Type5, class ID5, class Type6, class ID6, class Type7, class ID7, class Type8, class ID8>
	struct make<Type1, ID1, Type2, ID2, Type3, ID3, Type4, ID4, Type5, ID5, Type6, ID6, Type7, ID7, Type8, ID8>
	{
		typedef multi_vector<Type1, ID1, VectorBinder, typename multi_vector_t::template make<Type2, ID2, Type3, ID3, Type4, ID4, Type5, ID5, Type6, ID6, Type7, ID7, Type8, ID8>::type> type;
	};
};

} // namespace

using containers::multi_vector;
using containers::multi_vector_t;
using containers::append_swizzled;
using containers::swap;

} // namespace

#endif