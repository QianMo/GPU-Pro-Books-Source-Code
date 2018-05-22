///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2007-02-21
// Updated : 2007-02-21
// Licence : This source is under MIT License
// File    : glm/gtx/vecx.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cassert>

namespace glm
{
namespace detail{

	template <int N> const typename _bvecxGTX<N>::size_type _bvecxGTX<N>::value_size = N;

    // Bool constructors
	template <int N>
	inline _bvecxGTX<N>::_bvecxGTX()
	{
		for(int i = 0; i < N; ++i)
	        this->data[i] = false;
	}

	template <int N>
	inline _bvecxGTX<N>::_bvecxGTX(const _bvecxGTX<N>& v)
	{
		for(int i = 0; i < N; ++i)
	        this->data[i] = v[i];
	}

	template <int N>
    inline _bvecxGTX<N>::_bvecxGTX(const bool s)
	{
		for(int i = 0; i < N; ++i)
	        this->data[i] = s;
	}

	// Accesses
	template <int N>
    inline bool& _bvecxGTX<N>::operator[](int i)
	{
        assert(i >= 0 && i < N);
		return this->data[i];
	}

	template <int N>
    inline bool _bvecxGTX<N>::operator[](int i) const
	{
        assert(i >= 0 && i < N);
		return this->data[i];
	}

	template <int N>
    inline _bvecxGTX<N>::operator bool*()
	{
		return data;
	}

	template <int N>
    inline _bvecxGTX<N>::operator const bool*() const
	{
		return data;
	}

    // Operators
	template <int N>
    inline _bvecxGTX<N>& _bvecxGTX<N>::operator=(const _bvecxGTX<N>& v)
	{
		for(int i = 0; i < N; ++i)
			this->data[i] = v[i];
		return *this;
	}

	template <int N>
    inline _bvecxGTX<N> _bvecxGTX<N>::operator! () const
	{
		_bvecxGTX<N> result;
		for(int i = 0; i < N; ++i)
			result[i] = !this->data[i];
		return result;
	}

	template <int N, typename T> const typename _xvecxGTX<N, T>::size_type _xvecxGTX<N, T>::value_size = N;

	// Common constructors
	template <int N, typename T>
	inline _xvecxGTX<N, T>::_xvecxGTX()
	{
		for(int i = 0; i < N; ++i)
	        this->data[i] = T(0);
	}

	template <int N, typename T>
    inline _xvecxGTX<N, T>::_xvecxGTX(const _xvecxGTX<N, T>& v)
	{
		for(int i = 0; i < N; ++i)
	        this->data[i] = v[i];
	}

    // T constructors
	template <int N, typename T> 
    inline _xvecxGTX<N, T>::_xvecxGTX(const T s)
	{
		for(int i = 0; i < N; ++i)
	        this->data[i] = s;
	}

	// Accesses
    template <int N, typename T> 
    inline T& _xvecxGTX<N, T>::operator[](int i)
    {
        assert(i >= 0 && i < N);
		return this->data[i];
    }

    template <int N, typename T> 
    inline T _xvecxGTX<N, T>::operator[](int i) const
    {
		assert(i >= 0 && i < N);
        return this->data[i];
    }

    template <int N, typename T> 
    inline _xvecxGTX<N, T>::operator T*()
    {
        return data;
    }

    template <int N, typename T> 
    inline _xvecxGTX<N, T>::operator const T*() const 
    {
        return data;
    }

    template <int N, typename T>
    inline _xvecxGTX<N, T>& _xvecxGTX<N, T>::operator=(const _xvecxGTX<N, T>& v)
    {
		for(int i = 0; i < N; ++i)
	        this->data[i] = v[i];
        return *this;
    }

    template <int N, typename T> 
    inline _xvecxGTX<N, T>& _xvecxGTX<N, T>::operator+= (const T s)
    {
		for(int i = 0; i < N; ++i)
	        this->data[i] += s;
	    return *this;
    }

    template <int N, typename T> 
    inline _xvecxGTX<N, T>& _xvecxGTX<N, T>::operator+=(const _xvecxGTX<N, T>& v)
    {
		for(int i = 0; i < N; ++i)
	        this->data[i] += v[i];
	    return *this;
    }

    template <int N, typename T> 
    inline _xvecxGTX<N, T>& _xvecxGTX<N, T>::operator-= (const T s)
    {
		for(int i = 0; i < N; ++i)
	        this->data[i] -= s;
	    return *this;
    }

    template <int N, typename T> 
    inline _xvecxGTX<N, T>& _xvecxGTX<N, T>::operator-=(const _xvecxGTX<N, T>& v)
    {
		for(int i = 0; i < N; ++i)
	        this->data[i] -= v[i];
	    return *this;
    }

    template <int N, typename T> 
    inline _xvecxGTX<N, T>& _xvecxGTX<N, T>::operator*=(const T s)
    {
		for(int i = 0; i < N; ++i)
	        this->data[i] *= s;
	    return *this;
    }

    template <int N, typename T> 
    inline _xvecxGTX<N, T>& _xvecxGTX<N, T>::operator*= (const _xvecxGTX<N, T>& v)
    {
		for(int i = 0; i < N; ++i)
	        this->data[i] *= v[i];
	    return *this;
    }

    template <int N, typename T> 
    inline _xvecxGTX<N, T>& _xvecxGTX<N, T>::operator/=(const T s)
    {
		for(int i = 0; i < N; ++i)
	        this->data[i] /= s;
	    return *this;
    }

    template <int N, typename T> 
    inline _xvecxGTX<N, T>& _xvecxGTX<N, T>::operator/= (const _xvecxGTX<N, T>& v)
    {
		for(int i = 0; i < N; ++i)
	        this->data[i] /= v[i];
	    return *this;
    }


   // Unary constant operators
    template <int N, typename T> 
    inline const detail::_xvecxGTX<N, T> operator- (const detail::_xvecxGTX<N, T>& v)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = -v[i];
	    return result;
    }

    template <int N, typename T>
    inline const detail::_xvecxGTX<N, T> operator++ (const detail::_xvecxGTX<N, T>& v, int)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = v[i] + T(1);
	    return result;
    }

    template <int N, typename T> 
    inline const detail::_xvecxGTX<N, T> operator-- (const detail::_xvecxGTX<N, T>& v, int)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = v[i] - T(1);
	    return result;
    }

    // Binary operators
    template <int N, typename T>
	inline detail::_xvecxGTX<N, T> operator+ (const detail::_xvecxGTX<N, T>& v, const T s)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = v[i] + s;
	    return result;
    }

    template <int N, typename T>
    inline detail::_xvecxGTX<N, T> operator+ (const T s, const detail::_xvecxGTX<N, T>& v)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = v[i] + s;
	    return result;
    }

    template <int N, typename T>
    inline detail::_xvecxGTX<N, T> operator+ (const detail::_xvecxGTX<N, T>& v1, const detail::_xvecxGTX<N, T>& v2)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = v1[i] + v2[i];
	    return result;
    }
    
    template <int N, typename T>
	inline detail::_xvecxGTX<N, T> operator- (const detail::_xvecxGTX<N, T>& v, const T s)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = v[i] - s;
	    return result;
    }

    template <int N, typename T>
    inline detail::_xvecxGTX<N, T> operator- (const T s, const detail::_xvecxGTX<N, T>& v)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = s - v[i];
	    return result;
    }

    template <int N, typename T>
    inline detail::_xvecxGTX<N, T> operator- (const detail::_xvecxGTX<N, T>& v1, const detail::_xvecxGTX<N, T>& v2)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = v1[i] - v2[i];
	    return result;
    }

    template <int N, typename T>
    inline detail::_xvecxGTX<N, T> operator* (const detail::_xvecxGTX<N, T>& v, const T s)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = v[i] * s;
	    return result;
    }

    template <int N, typename T>
    inline detail::_xvecxGTX<N, T> operator* (const T s, const detail::_xvecxGTX<N, T>& v)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = s * v[i];
	    return result;
    }

    template <int N, typename T>
    inline detail::_xvecxGTX<N, T> operator* (const detail::_xvecxGTX<N, T>& v1, const detail::_xvecxGTX<N, T>& v2)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = v1[i] * v2[i];
	    return result;
    }

    template <int N, typename T>
    inline detail::_xvecxGTX<N, T> operator/ (const detail::_xvecxGTX<N, T>& v, const T s)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = v[i] / s;
	    return result;
    }

    template <int N, typename T>
    inline detail::_xvecxGTX<N, T> operator/ (const T s, const detail::_xvecxGTX<N, T>& v)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = s / v[i];
	    return result;
    }

    template <int N, typename T>
    inline detail::_xvecxGTX<N, T> operator/ (const detail::_xvecxGTX<N, T>& v1, const detail::_xvecxGTX<N, T>& v2)
    {
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = v1[i] / v2[i];
	    return result;
    }

}//namespace detail

	namespace gtx{
	namespace vecx{

	// Trigonometric Functions
	template <int N, typename T> 
	detail::_xvecxGTX<N, T> radiansGTX(const detail::_xvecxGTX<N, T>& degrees)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = radians(degrees[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> degreesGTX(const detail::_xvecxGTX<N, T>& radians)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = degrees(radians[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> sinGTX(const detail::_xvecxGTX<N, T>& angle)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = sin(angle[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> cosGTX(const detail::_xvecxGTX<N, T>& angle)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = cos(angle[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> tanGTX(const detail::_xvecxGTX<N, T>& angle)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = tan(angle[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> asinGTX(const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = asin(x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> acosGTX(const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = acos(x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> atanGTX(const detail::_xvecxGTX<N, T>& y, const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = atan(y[i], x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> atanGTX(const detail::_xvecxGTX<N, T>& y_over_x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = atan(y_over_x[i]);
	    return result;
	}

	// Exponential Functions
	template <int N, typename T> 
	detail::_xvecxGTX<N, T> powGTX(const detail::_xvecxGTX<N, T>& x, const detail::_xvecxGTX<N, T>& y)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = pow(x[i], y[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> expGTX(const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = exp(x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> logGTX(const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = log(x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> exp2GTX(const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = exp2(x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> log2GTX(const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = log2(x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> sqrtGTX(const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = sqrt(x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> inversesqrtGTX(const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = inversesqrt(x[i]);
	    return result;
	}

	// Common Functions
	template <int N, typename T> 
	detail::_xvecxGTX<N, T> absGTX(const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = abs(x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> signGTX(const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = sign(x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> floorGTX(const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = floor(x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> ceilGTX(const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = ceil(x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> fractGTX(const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = fract(x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> modGTX(const detail::_xvecxGTX<N, T>& x, T y)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = mod(x[i], y);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> modGTX(
		const detail::_xvecxGTX<N, T>& x, 
		const detail::_xvecxGTX<N, T>& y)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = mod(x[i], y[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> minGTX(
		const detail::_xvecxGTX<N, T>& x, 
		T y)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = min(x[i], y);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> minGTX(
		const detail::_xvecxGTX<N, T>& x, 
		const detail::_xvecxGTX<N, T>& y)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = min(x[i], y[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> maxGTX(
		const detail::_xvecxGTX<N, T>& x, 
		T y)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = max(x[i], y);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> maxGTX(
		const detail::_xvecxGTX<N, T>& x, 
		const detail::_xvecxGTX<N, T>& y)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = max(x[i], y[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> clampGTX(
		const detail::_xvecxGTX<N, T>& x, 
		T minVal, 
		T maxVal)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = clamp(x[i], minVal, maxVal);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> clampGTX(
		const detail::_xvecxGTX<N, T>& x, 
		const detail::_xvecxGTX<N, T>& minVal, 
		const detail::_xvecxGTX<N, T>& maxVal)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = clamp(x[i], minVal[i], maxVal[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> stepGTX(
		T edge, 
		const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = step(edge, x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> stepGTX(
		const detail::_xvecxGTX<N, T>& edge, 
		const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = step(edge[i], x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> smoothstepGTX(
		T edge0, 
		T edge1, 
		const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = step(edge0, edge1, x[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> smoothstepGTX(
		const detail::_xvecxGTX<N, T>& edge0, 
		const detail::_xvecxGTX<N, T>& edge1, 
		const detail::_xvecxGTX<N, T>& x)
	{
		detail::_xvecxGTX<N, T> result;
		for(int i = 0; i< N; ++i)
			result[i] = step(edge0[i], edge1[i], x[i]);
	    return result;
	}

	// Geometric Functions
	template <int N, typename T> 
	T lengthGTX(
		const detail::_xvecxGTX<N, T>& x)
	{
        T sqr = dot(x, x);
        return sqrt(sqr);
	}

	template <int N, typename T> 
	T distanceGTX(
		const detail::_xvecxGTX<N, T>& p0, 
		const detail::_xvecxGTX<N, T>& p1)
	{
        return lengthGTX(p1 - p0);
	}

	template <int N, typename T> 
	T dotGTX(
		const detail::_xvecxGTX<N, T>& x, 
		const detail::_xvecxGTX<N, T>& y)
	{
		T result = T(0);
		for(int i = 0; i < N; ++i)
			result += x[i] * y[i];
		return result;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> normalizeGTX(
		const detail::_xvecxGTX<N, T>& x)
	{
        T sqr = dot(x, x);
	    return x * inversesqrt(sqr);
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> faceforwardGTX(
		const detail::_xvecxGTX<N, T>& Normal, 
		const detail::_xvecxGTX<N, T>& I, 
		const detail::_xvecxGTX<N, T>& Nref)
	{
		return dot(Nref, I) < T(0) ? Normal : -Normal;
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> reflectGTX(
		const detail::_xvecxGTX<N, T>& I, 
		const detail::_xvecxGTX<N, T>& Normal)
	{
		return I - Normal * dot(Normal, I) * T(2);
	}

	template <int N, typename T> 
	detail::_xvecxGTX<N, T> refractGTX(
		const detail::_xvecxGTX<N, T>& I, 
		const detail::_xvecxGTX<N, T>& Normal, 
		T eta)
	{
        T dot = dot(Normal, I);
        T k = T(1) - eta * eta * (T(1) - dot * dot);
        if(k < T(0))
            return detail::_xvecxGTX<N, T>(T(0));
        else
            return eta * I - (eta * dot + sqrt(k)) * Normal;
	}

	// Vector Relational Functions
	template <int N, typename T> 
	detail::_bvecxGTX<N> lessThanGTX(
		const detail::_xvecxGTX<N, T>& x, 
		const detail::_xvecxGTX<N, T>& y)
	{
		detail::_bvecxGTX<N> result;
		for(int i = 0; i< N; ++i)
			result[i] = lessThan(x[i], y[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_bvecxGTX<N> lessThanEqualGTX(
		const detail::_xvecxGTX<N, T>& x, 
		const detail::_xvecxGTX<N, T>& y)
	{
		detail::_bvecxGTX<N> result;
		for(int i = 0; i< N; ++i)
			result[i] = lessThanEqual(x[i], y[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_bvecxGTX<N> greaterThanGTX(
		const detail::_xvecxGTX<N, T>& x, 
		const detail::_xvecxGTX<N, T>& y)
	{
		detail::_bvecxGTX<N> result;
		for(int i = 0; i< N; ++i)
			result[i] = greaterThan(x[i], y[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_bvecxGTX<N> greaterThanEqualGTX(
		const detail::_xvecxGTX<N, T>& x, 
		const detail::_xvecxGTX<N, T>& y)
	{
		detail::_bvecxGTX<N> result;
		for(int i = 0; i< N; ++i)
			result[i] = greaterThanEqual(x[i], y[i]);
	    return result;
	}

	template <int N> 
	detail::_bvecxGTX<N> equalGTX(
		const detail::_bvecxGTX<N>& x, 
		const detail::_bvecxGTX<N>& y)
	{
		detail::_bvecxGTX<N> result;
		for(int i = 0; i< N; ++i)
			result[i] = equal(x[i], y[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_bvecxGTX<N> equalGTX(
		const detail::_xvecxGTX<N, T>& x, 
		const detail::_xvecxGTX<N, T>& y)
	{
		detail::_bvecxGTX<N> result;
		for(int i = 0; i< N; ++i)
			result[i] = equal(x[i], y[i]);
	    return result;
	}

	template <int N> 
	detail::_bvecxGTX<N> notEqualGTX(
		const detail::_bvecxGTX<N>& x, 
		const detail::_bvecxGTX<N>& y)
	{
		detail::_bvecxGTX<N> result;
		for(int i = 0; i< N; ++i)
			result[i] = equal(x[i], y[i]);
	    return result;
	}

	template <int N, typename T> 
	detail::_bvecxGTX<N> notEqualGTX(
		const detail::_xvecxGTX<N, T>& x, 
		const detail::_xvecxGTX<N, T>& y)
	{
		detail::_bvecxGTX<N> result;
		for(int i = 0; i< N; ++i)
			result[i] = notEqual(x[i], y[i]);
	    return result;
	}

	template <int N> 
	bool anyGTX(const detail::_bvecxGTX<N>& x)
	{
		for(int i = 0; i< N; ++i)
			if(x[i]) return true;
	    return false;
	}

	template <int N> 
	bool allGTX(const detail::_bvecxGTX<N>& x)
	{
		for(int i = 0; i< N; ++i)
			if(!x[i]) return false;
	    return true;
	}

    template <int N> 
	detail::_bvecxGTX<N> notGTX(
		const detail::_bvecxGTX<N>& v)
	{
		detail::_bvecxGTX<N> result;
		for(int i = 0; i< N; ++i)
			result[i] = !v[i];
	    return result;
	}

	}//namespace vecx
	}//namespace gtx

} //namespace glm
