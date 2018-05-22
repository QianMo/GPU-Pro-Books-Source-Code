///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-08-18
// Updated : 2008-09-09
// Licence : This source is under MIT License
// File    : glm/core/type_tvec2.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	namespace detail
	{
		template <typename valType>
		typename tvec2<valType>::size_type tvec2<valType>::value_size()
		{
			return typename tvec2<valType>::size_type(2);
		}

		template <typename valType> 
		bool tvec2<valType>::is_vector()
		{
			return true;
		}

		//////////////////////////////////////
		// Accesses

		template <typename valType>
		inline valType& tvec2<valType>::operator[](typename tvec2<valType>::size_type i)
		{
			assert( i >= typename tvec2<valType>::size_type(0) && 
					i < tvec2<valType>::value_size());
			return (&x)[i];
		}

		template <typename valType>
		inline valType const & tvec2<valType>::operator[](typename tvec2<valType>::size_type i) const
		{
			assert( i >= typename tvec2<valType>::size_type(0) && 
					i < tvec2<valType>::value_size());
			return (&x)[i];
		}

		//////////////////////////////////////
		// Implicit basic constructors

		template <typename Type>
		inline tvec2<Type>::tvec2() :
			x(Type(0)),
			y(Type(0))
		{}

		template <typename Type>
		inline tvec2<Type>::tvec2(tvec2<Type> const & v) :
			x(v.x),
			y(v.y)
		{}

		//////////////////////////////////////
		// Explicit basic constructors

		template <typename valType>
		inline tvec2<valType>::tvec2(valType s) :
			x(s),
			y(s)
		{}

		template <typename valType>
		inline tvec2<valType>::tvec2(valType s1, valType s2) :
			x(s1),
			y(s2)
		{}

		//////////////////////////////////////
		// Swizzle constructors

		template <typename Type>
		inline tvec2<Type>::tvec2(tref2<Type> const & r) :
			x(r.x),
			y(r.y)
		{}

		//////////////////////////////////////
		// Convertion scalar constructors
		
		template <typename valType>
		template <typename U> 
		inline tvec2<valType>::tvec2(U x) :
			x(valType(x)),
			y(valType(x))
		{}

		template <typename valType>
		template <typename U, typename V> 
		inline tvec2<valType>::tvec2(U x, V y) :
			x(valType(x)),
			y(valType(y))
		{}

		//////////////////////////////////////
		// Convertion vector constructors

		template <typename valType>
		template <typename U> 
		inline tvec2<valType>::tvec2(tvec2<U> const & v) :
			x(valType(v.x)),
			y(valType(v.y))
		{}

		template <typename valType>
		template <typename U> 
		inline tvec2<valType>::tvec2(tvec3<U> const & v) :
			x(valType(v.x)),
			y(valType(v.y))
		{}

		template <typename valType>
		template <typename U> 
		inline tvec2<valType>::tvec2(tvec4<U> const & v) :
			x(valType(v.x)),
			y(valType(v.y))
		{}

		//////////////////////////////////////
		// Unary arithmetic operators

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator= (tvec2<valType> const & v)
		{
			this->x = v.x;
			this->y = v.y;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator+=(valType const & s)
		{
			this->x += s;
			this->y += s;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator+=(tvec2<valType> const & v)
		{
			this->x += v.x;
			this->y += v.y;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator-=(valType const & s)
		{
			this->x -= s;
			this->y -= s;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator-=(tvec2<valType> const & v)
		{
			this->x -= v.x;
			this->y -= v.y;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator*=(valType const & s)
		{
			this->x *= s;
			this->y *= s;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator*=(tvec2<valType> const & v)
		{
			this->x *= v.x;
			this->y *= v.y;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator/=(valType const & s)
		{
			this->x /= s;
			this->y /= s;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator/=(tvec2<valType> const & v)
		{
			this->x /= v.x;
			this->y /= v.y;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator++()
		{
			++this->x;
			++this->y;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator--()
		{
			--this->x;
			--this->y;
			return *this;
		}

		//////////////////////////////////////
		// Unary bit operators

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator%=(valType const & s)
		{
			this->x %= s;
			this->y %= s;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator%=(tvec2<valType> const & v)
		{
			this->x %= v.x;
			this->y %= v.y;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator&=(valType const & s)
		{
			this->x &= s;
			this->y &= s;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator&=(tvec2<valType> const & v)
		{
			this->x &= v.x;
			this->y &= v.y;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator|=(valType const & s)
		{
			this->x |= s;
			this->y |= s;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator|=(tvec2<valType> const & v)
		{
			this->x |= v.x;
			this->y |= v.y;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator^=(valType const & s)
		{
			this->x ^= s;
			this->y ^= s;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator^=(tvec2<valType> const & v)
		{
			this->x ^= v.x;
			this->y ^= v.y;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator<<=(valType const & s)
		{
			this->x <<= s;
			this->y <<= s;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator<<=(tvec2<valType> const & v)
		{
			this->x <<= v.x;
			this->y <<= v.y;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator>>=(valType const & s)
		{
			this->x >>= s;
			this->y >>= s;
			return *this;
		}

		template <typename valType>
		inline tvec2<valType>& tvec2<valType>::operator>>=(tvec2<valType> const & v)
		{
			this->x >>= v.x;
			this->y >>= v.y;
			return *this;
		}

		//////////////////////////////////////
		// Swizzle operators

		template <typename valType>
		inline valType tvec2<valType>::swizzle(comp x) const
		{
			return (*this)[x];
		}

		template <typename valType>
		inline tvec2<valType> tvec2<valType>::swizzle(comp x, comp y) const
		{
			return tvec2<valType>(
				(*this)[x],
				(*this)[y]);
		}

		template <typename valType>
		inline tvec3<valType> tvec2<valType>::swizzle(comp x, comp y, comp z) const
		{
			return tvec3<valType>(
				(*this)[x],
				(*this)[y],
				(*this)[z]);
		}

		template <typename valType>
		inline tvec4<valType> tvec2<valType>::swizzle(comp x, comp y, comp z, comp w) const
		{
			return tvec4<valType>(
				(*this)[x],
				(*this)[y],
				(*this)[z],
				(*this)[w]);
		}

		template <typename valType>
		inline tref2<valType> tvec2<valType>::swizzle(comp x, comp y)
		{
			return tref2<valType>(
				(*this)[x],
				(*this)[y]);
		}

		//////////////////////////////////////
		// Binary arithmetic operators

		template <typename T> 
		inline tvec2<T> operator+ (tvec2<T> const & v, T const & s)
		{
			return tvec2<T>(
				v.x + s,
				v.y + s);
		}

		template <typename T> 
		inline tvec2<T> operator+ (T const & s, tvec2<T> const & v)
		{
			return tvec2<T>(
				s + v.x,
				s + v.y);
		}

		template <typename T> 
		inline tvec2<T> operator+ (tvec2<T> const & v1, tvec2<T> const & v2)
		{
			return tvec2<T>(
				v1.x + v2.x,
				v1.y + v2.y);
		}

		//operator-
		template <typename T> 
		inline tvec2<T> operator- (tvec2<T> const & v, T const & s)
		{
			return tvec2<T>(
				v.x - s,
				v.y - s);
		}

		template <typename T> 
		inline tvec2<T> operator- (T const & s, tvec2<T> const & v)
		{
			return tvec2<T>(
				s - v.x,
				s - v.y);
		}

		template <typename T> 
		inline tvec2<T> operator- (tvec2<T> const & v1, tvec2<T> const & v2)
		{
			return tvec2<T>(
				v1.x - v2.x,
				v1.y - v2.y);
		}

		//operator*
		template <typename T> 
		inline tvec2<T> operator* (tvec2<T> const & v, T const & s)
		{
			return tvec2<T>(
				v.x * s,
				v.y * s);
		}

		template <typename T> 
		inline tvec2<T> operator* (T const & s, tvec2<T> const & v)
		{
			return tvec2<T>(
				s * v.x,
				s * v.y);
		}

		template <typename T> 
		inline tvec2<T> operator* (tvec2<T> const & v1, tvec2<T> const & v2)
		{
			return tvec2<T>(
				v1.x * v2.x,
				v1.y * v2.y);
		}

		//operator/
		template <typename T> 
		inline tvec2<T> operator/ (tvec2<T> const & v, T const & s)
		{
			return tvec2<T>(
				v.x / s,
				v.y / s);
		}

		template <typename T> 
		inline tvec2<T> operator/ (T const & s, tvec2<T> const & v)
		{
			return tvec2<T>(
				s / v.x,
				s / v.y);
		}

		template <typename T> 
		inline tvec2<T> operator/ (tvec2<T> const & v1, tvec2<T> const & v2)
		{
			return tvec2<T>(
				v1.x / v2.x,
				v1.y / v2.y);
		}

		// Unary constant operators
		template <typename T> 
		inline tvec2<T> operator- (tvec2<T> const & v)
		{
			return tvec2<T>(
				-v.x, 
				-v.y);
		}

		template <typename T> 
		inline tvec2<T> operator++ (tvec2<T> const & v, int)
		{
			return tvec2<T>(
				v.x + T(1), 
				v.y + T(1));
		}

		template <typename T> 
		inline tvec2<T> operator-- (tvec2<T> const & v, int)
		{
			return tvec2<T>(
				v.x - T(1), 
				v.y - T(1));
		}

		//////////////////////////////////////
		// Binary bit operators

		template <typename T>
		inline tvec2<T> operator% (tvec2<T> const & v, T const & s)
		{
			return tvec2<T>(
				v.x % s,
				v.y % s);
		}

		template <typename T>
		inline tvec2<T> operator% (T const & s, tvec2<T> const & v)
		{
			return tvec2<T>(
				s % v.x,
				s % v.y);
		}

		template <typename T>
		inline tvec2<T> operator% (tvec2<T> const & v1, tvec2<T> const & v2)
		{
			return tvec2<T>(
				v1.x % v2.x,
				v1.y % v2.y);
		}

		template <typename T>
		inline tvec2<T> operator& (tvec2<T> const & v, T const & s)
		{
			return tvec2<T>(
				v.x & s,
				v.y & s);
		}

		template <typename T>
		inline tvec2<T> operator& (T const & s, tvec2<T> const & v)
		{
			return tvec2<T>(
				s & v.x,
				s & v.y);
		}

		template <typename T>
		inline tvec2<T> operator& (tvec2<T> const & v1, tvec2<T> const & v2)
		{
			return tvec2<T>(
				v1.x & v2.x,
				v1.y & v2.y);
		}

		template <typename T>
		inline tvec2<T> operator| (tvec2<T> const & v, T const & s)
		{
			return tvec2<T>(
				v.x | s,
				v.y | s);
		}

		template <typename T>
		inline tvec2<T> operator| (T const & s, tvec2<T> const & v)
		{
			return tvec2<T>(
				s | v.x,
				s | v.y);
		}

		template <typename T>
		inline tvec2<T> operator| (const tvec2<T>& v1, const tvec2<T>& v2)
		{
			return tvec2<T>(
				v1.x | v2.x,
				v1.y | v2.y);
		}
		
		template <typename T>
		inline tvec2<T> operator^ (const tvec2<T>& v, T const & s)
		{
			return tvec2<T>(
				v.x ^ s,
				v.y ^ s);
		}

		template <typename T>
		inline tvec2<T> operator^ (T const & s, const tvec2<T>& v)
		{
			return tvec2<T>(
				s ^ v.x,
				s ^ v.y);
		}

		template <typename T>
		inline tvec2<T> operator^ (const tvec2<T>& v1, const tvec2<T>& v2)
		{
			return tvec2<T>(
				v1.x ^ v2.x,
				v1.y ^ v2.y);
		}

		template <typename T>
		inline tvec2<T> operator<< (const tvec2<T>& v, T const & s)
		{
			return tvec2<T>(
				v.x << s,
				v.y << s);
		}

		template <typename T>
		inline tvec2<T> operator<< (T const & s, const tvec2<T>& v)
		{
			return tvec2<T>(
				s << v.x,
				s << v.y);
		}

		template <typename T>
		inline tvec2<T> operator<< (const tvec2<T>& v1, const tvec2<T>& v2)
		{
			return tvec2<T>(
				v1.x << v2.x,
				v1.y << v2.y);
		}

		template <typename T>
		inline tvec2<T> operator>> (const tvec2<T>& v, T const & s)
		{
			return tvec2<T>(
				v.x >> s,
				v.y >> s);
		}

		template <typename T>
		inline tvec2<T> operator>> (T const & s, const tvec2<T>& v)
		{
			return tvec2<T>(
				s >> v.x,
				s >> v.y);
		}

		template <typename T>
		inline tvec2<T> operator>> (const tvec2<T>& v1, const tvec2<T>& v2)
		{
			return tvec2<T>(
				v1.x >> v2.x,
				v1.y >> v2.y);
		}

		template <typename T> 
		inline tvec2<T> operator~ (const tvec2<T>& v)
		{
			return tvec2<T>(
				~v.x,
				~v.y);
		}

		//////////////////////////////////////
		// tref definition

		template <typename T> 
		tref2<T>::tref2(T& x, T& y) :
			x(x),
			y(y)
		{}

		template <typename T> 
		tref2<T>::tref2(const tref2<T>& r) :
			x(r.x),
			y(r.y)
		{}

		template <typename T> 
		tref2<T>::tref2(const tvec2<T>& v) :
			x(v.x),
			y(v.y)
		{}

		template <typename T> 
		tref2<T>& tref2<T>::operator= (const tref2<T>& r)
		{
			x = r.x;
			y = r.y;
			return *this;
		}

		template <typename T> 
		tref2<T>& tref2<T>::operator= (const tvec2<T>& v)
		{
			x = v.x;
			y = v.y;
			return *this;
		}

	}//namespace detail
}//namespace glm
