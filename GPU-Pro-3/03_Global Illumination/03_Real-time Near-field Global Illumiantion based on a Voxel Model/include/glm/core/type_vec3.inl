///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-08-22
// Updated : 2008-09-09
// Licence : This source is under MIT License
// File    : glm/core/type_tvec3.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	namespace detail
	{
		template <typename valType>
		typename tvec3<valType>::size_type tvec3<valType>::value_size()
		{
			return typename tvec3<valType>::size_type(3);
		}

		template <typename valType> 
		bool tvec3<valType>::is_vector()
		{
			return true;
		}

		//////////////////////////////////////
		// Accesses

		template <typename valType>
		inline valType& tvec3<valType>::operator[](typename tvec3<valType>::size_type i)
		{
			assert( i >= typename tvec3<valType>::size_type(0) && 
					i < tvec3<valType>::value_size());

			return (&x)[i];
		}

		template <typename valType>
		inline valType const & tvec3<valType>::operator[](typename tvec3<valType>::size_type i) const
		{
			assert( i >= typename tvec3<valType>::size_type(0) && 
					i < tvec3<valType>::value_size());

			return (&x)[i];
		}

		//////////////////////////////////////
		// Implicit basic constructors

		template <typename valType>
		inline tvec3<valType>::tvec3() :
			x(valType(0)),
			y(valType(0)),
			z(valType(0))
		{}

		template <typename valType>
		inline tvec3<valType>::tvec3(const tvec3<valType>& v) :
			x(v.x),
			y(v.y),
			z(v.z)
		{}

		//////////////////////////////////////
		// Explicit basic constructors

		template <typename valType>
		inline tvec3<valType>::tvec3(valType s) :
			x(s),
			y(s),
			z(s)
		{}

		template <typename valType>
		inline tvec3<valType>::tvec3(valType s0, valType s1, valType s2) :
			x(s0),
			y(s1),
			z(s2)
		{}

		//////////////////////////////////////
		// Swizzle constructors

		template <typename valType>
		inline tvec3<valType>::tvec3(const tref3<valType>& r) :
			x(r.x),
			y(r.y),
			z(r.z)
		{}

		//////////////////////////////////////
		// Convertion scalar constructors
		
		template <typename valType>
		template <typename U> 
		inline tvec3<valType>::tvec3(U x) :
			x(valType(x)),
			y(valType(x)),
			z(valType(x))
		{}

		template <typename valType>
		template <typename A, typename B, typename C> 
		inline tvec3<valType>::tvec3(A x, B y, C z) :
			x(valType(x)),
			y(valType(y)),
			z(valType(z))
		{}

		//////////////////////////////////////
		// Convertion vector constructors

		template <typename valType>
		template <typename A, typename B> 
		inline tvec3<valType>::tvec3(const tvec2<A>& v, B s) :
			x(valType(v.x)),
			y(valType(v.y)),
			z(valType(s))
		{}

		template <typename valType>
		template <typename A, typename B> 
		inline tvec3<valType>::tvec3(A s, const tvec2<B>& v) :
			x(valType(s)),
			y(valType(v.x)),
			z(valType(v.y))
		{}

		template <typename valType>
		template <typename U> 
		inline tvec3<valType>::tvec3(const tvec3<U>& v) :
			x(valType(v.x)),
			y(valType(v.y)),
			z(valType(v.z))
		{}

		template <typename valType>
		template <typename U> 
		inline tvec3<valType>::tvec3(const tvec4<U>& v) :
			x(valType(v.x)),
			y(valType(v.y)),
			z(valType(v.z))
		{}

		//////////////////////////////////////
		// Unary arithmetic operators

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator= (const tvec3<valType>& v)
		{
			this->x = v.x;
			this->y = v.y;
			this->z = v.z;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator+=(valType const & s)
		{
			this->x += s;
			this->y += s;
			this->z += s;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator+=(const tvec3<valType>& v)
		{
			this->x += v.x;
			this->y += v.y;
			this->z += v.z;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator-=(valType const & s)
		{
			this->x -= s;
			this->y -= s;
			this->z -= s;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator-=(const tvec3<valType>& v)
		{
			this->x -= v.x;
			this->y -= v.y;
			this->z -= v.z;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator*=(valType const & s)
		{
			this->x *= s;
			this->y *= s;
			this->z *= s;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator*=(const tvec3<valType>& v)
		{
			this->x *= v.x;
			this->y *= v.y;
			this->z *= v.z;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator/=(valType const & s)
		{
			this->x /= s;
			this->y /= s;
			this->z /= s;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator/=(const tvec3<valType>& v)
		{
			this->x /= v.x;
			this->y /= v.y;
			this->z /= v.z;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator++()
		{
			++this->x;
			++this->y;
			++this->z;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator--()
		{
			--this->x;
			--this->y;
			--this->z;
			return *this;
		}

		//////////////////////////////////////
		// Unary bit operators

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator%=(valType const & s)
		{
			this->x %= s;
			this->y %= s;
			this->z %= s;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator%=(const tvec3<valType>& v)
		{
			this->x %= v.x;
			this->y %= v.y;
			this->z %= v.z;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator&=(valType const & s)
		{
			this->x &= s;
			this->y &= s;
			this->z &= s;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator&=(const tvec3<valType>& v)
		{
			this->x &= v.x;
			this->y &= v.y;
			this->z &= v.z;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator|=(valType const & s)
		{
			this->x |= s;
			this->y |= s;
			this->z |= s;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator|=(const tvec3<valType>& v)
		{
			this->x |= v.x;
			this->y |= v.y;
			this->z |= v.z;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator^=(valType const & s)
		{
			this->x ^= s;
			this->y ^= s;
			this->z ^= s;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator^=(const tvec3<valType>& v)
		{
			this->x ^= v.x;
			this->y ^= v.y;
			this->z ^= v.z;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator<<=(valType const & s)
		{
			this->x <<= s;
			this->y <<= s;
			this->z <<= s;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator<<=(const tvec3<valType>& v)
		{
			this->x <<= v.x;
			this->y <<= v.y;
			this->z <<= v.z;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator>>=(valType const & s)
		{
			this->x >>= s;
			this->y >>= s;
			this->z >>= s;
			return *this;
		}

		template <typename valType>
		inline tvec3<valType>& tvec3<valType>::operator>>=(const tvec3<valType>& v)
		{
			this->x >>= v.x;
			this->y >>= v.y;
			this->z >>= v.z;
			return *this;
		}

		//////////////////////////////////////
		// Swizzle operators

		template <typename valType>
		inline valType tvec3<valType>::swizzle(comp x) const
		{
			return (*this)[x];
		}

		template <typename valType>
		inline tvec2<valType> tvec3<valType>::swizzle(comp x, comp y) const
		{
			return tvec2<valType>(
				(*this)[x],
				(*this)[y]);
		}

		template <typename valType>
		inline tvec3<valType> tvec3<valType>::swizzle(comp x, comp y, comp z) const
		{
			return tvec3<valType>(
				(*this)[x],
				(*this)[y],
				(*this)[z]);
		}

		template <typename valType>
		inline tvec4<valType> tvec3<valType>::swizzle(comp x, comp y, comp z, comp w) const
		{
			return tvec4<valType>(
				(*this)[x],
				(*this)[y],
				(*this)[z],
				(*this)[w]);
		}

		template <typename valType>
		inline tref3<valType> tvec3<valType>::swizzle(comp x, comp y, comp z)
		{
			return tref3<valType>(
				(*this)[x],
				(*this)[y],
				(*this)[z]);
		}

		//////////////////////////////////////
		// Binary arithmetic operators

		template <typename T> 
		inline tvec3<T> operator+ (const tvec3<T>& v, T const & s)
		{
			return tvec3<T>(
				v.x + s,
				v.y + s,
				v.z + s);
		}

		template <typename T> 
		inline tvec3<T> operator+ (T const & s, const tvec3<T>& v)
		{
			return tvec3<T>(
				s + v.x,
				s + v.y,
				s + v.z);
		}

		template <typename T> 
		inline tvec3<T> operator+ (const tvec3<T>& v1, const tvec3<T>& v2)
		{
			return tvec3<T>(
				v1.x + v2.x,
				v1.y + v2.y,
				v1.z + v2.z);
		}

		//operator-
		template <typename T> 
		inline tvec3<T> operator- (const tvec3<T>& v, T const & s)
		{
			return tvec3<T>(
				v.x - s,
				v.y - s,
				v.z - s);
		}

		template <typename T> 
		inline tvec3<T> operator- (T const & s, const tvec3<T>& v)
		{
			return tvec3<T>(
				s - v.x,
				s - v.y,
				s - v.z);
		}

		template <typename T> 
		inline tvec3<T> operator- (const tvec3<T>& v1, const tvec3<T>& v2)
		{
			return tvec3<T>(
				v1.x - v2.x,
				v1.y - v2.y,
				v1.z - v2.z);
		}

		//operator*
		template <typename T> 
		inline tvec3<T> operator* (const tvec3<T>& v, T const & s)
		{
			return tvec3<T>(
				v.x * s,
				v.y * s,
				v.z * s);
		}

		template <typename T> 
		inline tvec3<T> operator* (T const & s, const tvec3<T>& v)
		{
			return tvec3<T>(
				s * v.x,
				s * v.y,
				s * v.z);
		}

		template <typename T> 
		inline tvec3<T> operator* (const tvec3<T>& v1, const tvec3<T> & v2)
		{
			return tvec3<T>(
				v1.x * v2.x,
				v1.y * v2.y,
				v1.z * v2.z);
		}

		//operator/
		template <typename T> 
		inline tvec3<T> operator/ (const tvec3<T>& v, T const & s)
		{
			return tvec3<T>(
				v.x / s,
				v.y / s,
				v.z / s);
		}

		template <typename T> 
		inline tvec3<T> operator/ (T const & s, const tvec3<T>& v)
		{
			return tvec3<T>(
				s / v.x,
				s / v.y,
				s / v.z);
		}

		template <typename T> 
		inline tvec3<T> operator/ (const tvec3<T>& v1, const tvec3<T>& v2)
		{
			return tvec3<T>(
				v1.x / v2.x,
				v1.y / v2.y,
				v1.z / v2.z);
		}

		// Unary constant operators
		template <typename T> 
		inline tvec3<T> operator- (const tvec3<T>& v)
		{
			return tvec3<T>(
				-v.x, 
				-v.y, 
				-v.z);
		}

		template <typename T> 
		inline tvec3<T> operator++ (const tvec3<T>& v, int)
		{
			return tvec3<T>(
				v.x + T(1), 
				v.y + T(1), 
				v.z + T(1));
		}

		template <typename T> 
		inline tvec3<T> operator-- (const tvec3<T>& v, int)
		{
			return tvec3<T>(
				v.x - T(1), 
				v.y - T(1), 
				v.z - T(1));
		}

		//////////////////////////////////////
		// Binary bit operators

		template <typename T>
		inline tvec3<T> operator% (const tvec3<T>& v, T const & s)
		{
			return tvec3<T>(
				v.x % s,
				v.y % s,
				v.z % s);
		}

		template <typename T>
		inline tvec3<T> operator% (T const & s, const tvec3<T>& v)
		{
			return tvec3<T>(
				s % v.x,
				s % v.y,
				s % v.z);
		}

		template <typename T>
		inline tvec3<T> operator% (const tvec3<T>& v1, const tvec3<T>& v2)
		{
			return tvec3<T>(
				v1.x % v2.x,
				v1.y % v2.y,
				v1.z % v2.z);
		}

		template <typename T>
		inline tvec3<T> operator& (const tvec3<T>& v, T const & s)
		{
			return tvec3<T>(
				v.x & s,
				v.y & s,
				v.z & s);
		}

		template <typename T>
		inline tvec3<T> operator& (T const & s, const tvec3<T>& v)
		{
			return tvec3<T>(
				s & v.x,
				s & v.y,
				s & v.z);
		}

		template <typename T>
		inline tvec3<T> operator& (const tvec3<T>& v1, const tvec3<T>& v2)
		{
			return tvec3<T>(
				v1.x & v2.x,
				v1.y & v2.y,
				v1.z & v2.z);
		}

		template <typename T>
		inline tvec3<T> operator| (const tvec3<T>& v, T const & s)
		{
			return tvec3<T>(
				v.x | s,
				v.y | s,
				v.z | s);
		}

		template <typename T>
		inline tvec3<T> operator| (T const & s, const tvec3<T>& v)
		{
			return tvec3<T>(
				s | v.x,
				s | v.y,
				s | v.z);
		}

		template <typename T>
		inline tvec3<T> operator| (const tvec3<T>& v1, const tvec3<T>& v2)
		{
			return tvec3<T>(
				v1.x | v2.x,
				v1.y | v2.y,
				v1.z | v2.z);
		}
		
		template <typename T>
		inline tvec3<T> operator^ (const tvec3<T>& v, T const & s)
		{
			return tvec3<T>(
				v.x ^ s,
				v.y ^ s,
				v.z ^ s);
		}

		template <typename T>
		inline tvec3<T> operator^ (T const & s, const tvec3<T>& v)
		{
			return tvec3<T>(
				s ^ v.x,
				s ^ v.y,
				s ^ v.z);
		}

		template <typename T>
		inline tvec3<T> operator^ (const tvec3<T>& v1, const tvec3<T>& v2)
		{
			return tvec3<T>(
				v1.x ^ v2.x,
				v1.y ^ v2.y,
				v1.z ^ v2.z);
		}

		template <typename T>
		inline tvec3<T> operator<< (const tvec3<T>& v, T const & s)
		{
			return tvec3<T>(
				v.x << s,
				v.y << s,
				v.z << s);
		}

		template <typename T>
		inline tvec3<T> operator<< (T const & s, const tvec3<T>& v)
		{
			return tvec3<T>(
				s << v.x,
				s << v.y,
				s << v.z);
		}

		template <typename T>
		inline tvec3<T> operator<< (const tvec3<T>& v1, const tvec3<T>& v2)
		{
			return tvec3<T>(
				v1.x << v2.x,
				v1.y << v2.y,
				v1.z << v2.z);
		}

		template <typename T>
		inline tvec3<T> operator>> (const tvec3<T>& v, T const & s)
		{
			return tvec3<T>(
				v.x >> s,
				v.y >> s,
				v.z >> s);
		}

		template <typename T>
		inline tvec3<T> operator>> (T const & s, const tvec3<T>& v)
		{
			return tvec3<T>(
				s >> v.x,
				s >> v.y,
				s >> v.z);
		}

		template <typename T>
		inline tvec3<T> operator>> (const tvec3<T>& v1, const tvec3<T>& v2)
		{
			return tvec3<T>(
				v1.x >> v2.x,
				v1.y >> v2.y,
				v1.z >> v2.z);
		}

		template <typename T> 
		inline tvec3<T> operator~ (const tvec3<T>& v)
		{
			return tvec3<T>(
				~v.x,
				~v.y,
				~v.z);
		}

		//////////////////////////////////////
		// tref definition

		template <typename T> 
		tref3<T>::tref3(T& x, T& y, T& z) :
			x(x),
			y(y),
			z(z)
		{}

		template <typename T> 
		tref3<T>::tref3(const tref3<T>& r) :
			x(r.x),
			y(r.y),
			z(r.z)
		{}

		template <typename T> 
		tref3<T>::tref3(const tvec3<T>& v) :
			x(v.x),
			y(v.y),
			z(v.z)
		{}

		template <typename T> 
		tref3<T>& tref3<T>::operator= (const tref3<T>& r)
		{
			x = r.x;
			y = r.y;
			z = r.z;
			return *this;
		}

		template <typename T> 
		tref3<T>& tref3<T>::operator= (const tvec3<T>& v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			return *this;
		}

	}//namespace detail
}//namespace glm
