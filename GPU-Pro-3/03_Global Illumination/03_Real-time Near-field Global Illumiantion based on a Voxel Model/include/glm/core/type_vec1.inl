///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-08-25
// Updated : 2008-09-09
// Licence : This source is under MIT License
// File    : glm/core/type_vec1.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	namespace detail
	{
		template <typename valType>
		typename tvec1<valType>::size_type tvec1<valType>::value_size()
		{
			return typename tvec1<valType>::size_type(1);
		}

		template <typename valType> 
		bool tvec1<valType>::is_vector()
		{
			return true;
		}

		//////////////////////////////////////
		// Accesses

		template <typename valType>
		valType& tvec1<valType>::operator[](typename tvec1<valType>::size_type i)
		{
			assert( i >= typename tvec1<valType>::size_type(0) && 
					i < tvec1<valType>::value_size());
			return (&x)[i];
		}

		template <typename valType>
		valType const & tvec1<valType>::operator[](typename tvec1<valType>::size_type i) const
		{
			assert( i >= typename tvec1<valType>::size_type(0) && 
					i < tvec1<valType>::value_size());
			return (&x)[i];
		}

		//////////////////////////////////////
		// Implicit basic constructors

		template <typename valType>
		tvec1<valType>::tvec1() :
			x(valType(0))
		{}

		template <typename valType>
		tvec1<valType>::tvec1(const tvec1<valType>& v) :
			x(v.x)
		{}

		//////////////////////////////////////
		// Explicit basic constructors

		template <typename valType>
		tvec1<valType>::tvec1(valType s) :
			x(s)
		{}

		//////////////////////////////////////
		// Swizzle constructors

		template <typename valType>
		tvec1<valType>::tvec1(const tref1<valType>& r) :
			x(r.x)
		{}

		//////////////////////////////////////
		// Convertion scalar constructors
		
		template <typename valType>
		template <typename U> 
		tvec1<valType>::tvec1(U x) :
			x(valType(x))
		{}

		//////////////////////////////////////
		// Convertion vector constructors

		template <typename valType>
		template <typename U> 
		tvec1<valType>::tvec1(const tvec2<U>& v) :
			x(valType(v.x))
		{}

		template <typename valType>
		template <typename U> 
		tvec1<valType>::tvec1(const tvec3<U>& v) :
			x(valType(v.x))
		{}

		template <typename valType>
		template <typename U> 
		tvec1<valType>::tvec1(const tvec4<U>& v) :
			x(valType(v.x))
		{}

		//////////////////////////////////////
		// Unary arithmetic operators

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator= (const tvec1<valType>& v)
		{
			this->x = v.x;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator+=(valType const & s)
		{
			this->x += s;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator+=(const tvec1<valType>& v)
		{
			this->x += v.x;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator-=(valType const & s)
		{
			this->x -= s;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator-=(const tvec1<valType>& v)
		{
			this->x -= v.x;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator*=(valType const & s)
		{
			this->x *= s;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator*=(const tvec1<valType>& v)
		{
			this->x *= v.x;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator/=(valType const & s)
		{
			this->x /= s;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator/=(const tvec1<valType>& v)
		{
			this->x /= v.x;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator++()
		{
			++this->x;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator--()
		{
			--this->x;
			return *this;
		}

		//////////////////////////////////////
		// Unary bit operators

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator%=(valType const & s)
		{
			this->x %= s;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator%=(const tvec1<valType>& v)
		{
			this->x %= v.x;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator&=(valType const & s)
		{
			this->x &= s;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator&=(const tvec1<valType>& v)
		{
			this->x &= v.x;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator|=(valType const & s)
		{
			this->x |= s;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator|=(const tvec1<valType>& v)
		{
			this->x |= v.x;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator^=(valType const & s)
		{
			this->x ^= s;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator^=(const tvec1<valType>& v)
		{
			this->x ^= v.x;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator<<=(valType const & s)
		{
			this->x <<= s;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator<<=(const tvec1<valType>& v)
		{
			this->x <<= v.x;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator>>=(valType const & s)
		{
			this->x >>= s;
			return *this;
		}

		template <typename valType>
		tvec1<valType>& tvec1<valType>::operator>>=(const tvec1<valType>& v)
		{
			this->x >>= v.x;
			return *this;
		}

		//////////////////////////////////////
		// Swizzle operators

		template <typename valType>
		valType tvec1<valType>::swizzle(comp x) const
		{
			return (*this)[x];
		}

		template <typename valType>
		tvec2<valType> tvec1<valType>::swizzle(comp x, comp y) const
		{
			return tvec2<valType>(
				(*this)[x],
				(*this)[y]);
		}

		template <typename valType>
		tvec3<valType> tvec1<valType>::swizzle(comp x, comp y, comp z) const
		{
			return tvec3<valType>(
				(*this)[x],
				(*this)[y],
				(*this)[z]);
		}

		template <typename valType>
		tvec4<valType> tvec1<valType>::swizzle(comp x, comp y, comp z, comp w) const
		{
			return tvec4<valType>(
				(*this)[x],
				(*this)[y],
				(*this)[z],
				(*this)[w]);
		}

		template <typename valType>
		tref1<valType> tvec1<valType>::swizzle(comp x)
		{
			return tref1<valType>(
				(*this)[x]);
		}

		//////////////////////////////////////
		// Binary arithmetic operators

		template <typename T> 
		inline tvec1<T> operator+ (const tvec1<T>& v, T const & s)
		{
			return tvec1<T>(
				v.x + s);
		}

		template <typename T> 
		inline tvec1<T> operator+ (T const & s, const tvec1<T>& v)
		{
			return tvec1<T>(
				s + v.x);
		}

		template <typename T> 
		inline tvec1<T> operator+ (const tvec1<T>& v1, const tvec1<T>& v2)
		{
			return tvec1<T>(
				v1.x + v2.x);
		}

		//operator-
		template <typename T> 
		inline tvec1<T> operator- (const tvec1<T>& v, T const & s)
		{
			return tvec1<T>(
				v.x - s);
		}

		template <typename T> 
		inline tvec1<T> operator- (T const & s, const tvec1<T>& v)
		{
			return tvec1<T>(
				s - v.x);
		}

		template <typename T> 
		inline tvec1<T> operator- (const tvec1<T>& v1, const tvec1<T>& v2)
		{
			return tvec1<T>(
				v1.x - v2.x);
		}

		//operator*
		template <typename T> 
		inline tvec1<T> operator* (const tvec1<T>& v, T const & s)
		{
			return tvec1<T>(
				v.x * s);
		}

		template <typename T> 
		inline tvec1<T> operator* (T const & s, const tvec1<T>& v)
		{
			return tvec1<T>(
				s * v.x);
		}

		template <typename T> 
		inline tvec1<T> operator* (const tvec1<T>& v1, const tvec1<T> & v2)
		{
			return tvec1<T>(
				v1.x * v2.x);
		}

		//operator/
		template <typename T> 
		inline tvec1<T> operator/ (const tvec1<T>& v, T const & s)
		{
			return tvec1<T>(
				v.x / s);
		}

		template <typename T> 
		inline tvec1<T> operator/ (T const & s, const tvec1<T>& v)
		{
			return tvec1<T>(
				s / v.x);
		}

		template <typename T> 
		inline tvec1<T> operator/ (const tvec1<T>& v1, const tvec1<T>& v2)
		{
			return tvec1<T>(
				v1.x / v2.x);
		}

		// Unary constant operators
		template <typename T> 
		inline tvec1<T> operator- (const tvec1<T>& v)
		{
			return tvec1<T>(
				-v.x);
		}

		template <typename T> 
		inline tvec1<T> operator++ (const tvec1<T>& v, int)
		{
			return tvec1<T>(
				v.x + T(1));
		}

		template <typename T> 
		inline tvec1<T> operator-- (const tvec1<T>& v, int)
		{
			return tvec1<T>(
				v.x - T(1));
		}

		//////////////////////////////////////
		// Binary bit operators

		template <typename T>
		inline tvec1<T> operator% (const tvec1<T>& v, T const & s)
		{
			return tvec1<T>(
				v.x % s);
		}

		template <typename T>
		inline tvec1<T> operator% (T const & s, const tvec1<T>& v)
		{
			return tvec1<T>(
				s % v.x);
		}

		template <typename T>
		inline tvec1<T> operator% (const tvec1<T>& v1, const tvec1<T>& v2)
		{
			return tvec1<T>(
				v1.x % v2.x);
		}

		template <typename T>
		inline tvec1<T> operator& (const tvec1<T>& v, T const & s)
		{
			return tvec1<T>(
				v.x & s);
		}

		template <typename T>
		inline tvec1<T> operator& (T const & s, const tvec1<T>& v)
		{
			return tvec1<T>(
				s & v.x);
		}

		template <typename T>
		inline tvec1<T> operator& (const tvec1<T>& v1, const tvec1<T>& v2)
		{
			return tvec1<T>(
				v1.x & v2.x);
		}

		template <typename T>
		inline tvec1<T> operator| (const tvec1<T>& v, T const & s)
		{
			return tvec1<T>(
				v.x | s);
		}

		template <typename T>
		inline tvec1<T> operator| (T const & s, const tvec1<T>& v)
		{
			return tvec1<T>(
				s | v.x);
		}

		template <typename T>
		inline tvec1<T> operator| (const tvec1<T>& v1, const tvec1<T>& v2)
		{
			return tvec1<T>(
				v1.x | v2.x);
		}
		
		template <typename T>
		inline tvec1<T> operator^ (const tvec1<T>& v, T const & s)
		{
			return tvec1<T>(
				v.x ^ s);
		}

		template <typename T>
		inline tvec1<T> operator^ (T const & s, const tvec1<T>& v)
		{
			return tvec1<T>(
				s ^ v.x);
		}

		template <typename T>
		inline tvec1<T> operator^ (const tvec1<T>& v1, const tvec1<T>& v2)
		{
			return tvec1<T>(
				v1.x ^ v2.x);
		}

		template <typename T>
		inline tvec1<T> operator<< (const tvec1<T>& v, T const & s)
		{
			return tvec1<T>(
				v.x << s);
		}

		template <typename T>
		inline tvec1<T> operator<< (T const & s, const tvec1<T>& v)
		{
			return tvec1<T>(
				s << v.x);
		}

		template <typename T>
		inline tvec1<T> operator<< (const tvec1<T>& v1, const tvec1<T>& v2)
		{
			return tvec1<T>(
				v1.x << v2.x);
		}

		template <typename T>
		inline tvec1<T> operator>> (const tvec1<T>& v, T const & s)
		{
			return tvec1<T>(
				v.x >> s);
		}

		template <typename T>
		inline tvec1<T> operator>> (T const & s, const tvec1<T>& v)
		{
			return tvec1<T>(
				s >> v.x);
		}

		template <typename T>
		inline tvec1<T> operator>> (const tvec1<T>& v1, const tvec1<T>& v2)
		{
			return tvec1<T>(
				v1.x >> v2.x);
		}

		template <typename T> 
		inline tvec1<T> operator~ (const tvec1<T>& v)
		{
			return tvec1<T>(
				~v.x);
		}

		//////////////////////////////////////
		// tref definition

		template <typename T> 
		tref1<T>::tref1(T& x) :
			x(x)
		{}

		template <typename T> 
		tref1<T>::tref1(const tref1<T>& r) :
			x(r.x)
		{}

		template <typename T> 
		tref1<T>::tref1(const tvec1<T>& v) :
			x(v.x)
		{}

		template <typename T> 
		tref1<T>& tref1<T>::operator= (const tref1<T>& r)
		{
			x = r.x;
			return *this;
		}

		template <typename T> 
		tref1<T>& tref1<T>::operator= (const tvec1<T>& v)
		{
			x = v.x;
			return *this;
		}

	}//namespace detail
}//namespace glm
