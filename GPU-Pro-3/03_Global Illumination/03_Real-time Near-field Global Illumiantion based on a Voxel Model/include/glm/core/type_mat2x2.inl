///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2005-01-16
// Updated : 2007-03-01
// Licence : This source is under MIT License
// File    : glm/core/type_mat2x2.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail{

	template <typename valType> 
	typename tmat2x2<valType>::size_type tmat2x2<valType>::value_size()
	{
		return typename tmat2x2<valType>::size_type(2);
	}

	template <typename valType> 
	typename tmat2x2<valType>::size_type tmat2x2<valType>::col_size()
	{
		return typename tmat2x2<valType>::size_type(2);
	}

	template <typename valType> 
	typename tmat2x2<valType>::size_type tmat2x2<valType>::row_size()
	{
		return typename tmat2x2<valType>::size_type(2);
	}

	template <typename valType> 
	bool tmat2x2<valType>::is_matrix()
	{
		return true;
	}

	//////////////////////////////////////
	// Accesses

	template <typename valType>
	detail::tvec2<valType>& tmat2x2<valType>::operator[](typename tmat2x2<valType>::size_type i)
	{
		assert(
			i >= typename tmat2x2<valType>::size_type(0) && 
			i < tmat2x2<valType>::col_size());

		return value[i];
	}

	template <typename valType>
	const detail::tvec2<valType>& tmat2x2<valType>::operator[](typename tmat2x2<valType>::size_type i) const
	{
		assert(
			i >= typename tmat2x2<valType>::size_type(0) && 
			i < tmat2x2<valType>::col_size());

		return value[i];
	}

    //////////////////////////////////////////////////////////////
    // mat2 constructors

    template <typename T> 
    inline tmat2x2<T>::tmat2x2()
    {
        this->value[0] = detail::tvec2<T>(1, 0);
        this->value[1] = detail::tvec2<T>(0, 1);
    }

    template <typename T> 
    inline tmat2x2<T>::tmat2x2(const tmat2x2<T> & m)
    {
        this->value[0] = m.value[0];
        this->value[1] = m.value[1];
    }

    template <typename T> 
    inline tmat2x2<T>::tmat2x2(const T f)
    {
        this->value[0] = detail::tvec2<T>(f, 0);
        this->value[1] = detail::tvec2<T>(0, f);
    }

    template <typename T> 
    inline tmat2x2<T>::tmat2x2(const T x0, const T y0, const T x1, const T y1)
    {
        this->value[0] = detail::tvec2<T>(x0, y0);
        this->value[1] = detail::tvec2<T>(x1, y1);
    }

    template <typename T> 
    inline tmat2x2<T>::tmat2x2(const detail::tvec2<T>& v0, const detail::tvec2<T>& v1)
    {
        this->value[0] = v0;
        this->value[1] = v1;
    }

    //////////////////////////////////////////////////////////////
    // mat2 conversions

    template <typename T> 
    template <typename U> 
    inline tmat2x2<T>::tmat2x2(const tmat2x2<U>& m)
    {
        this->value[0] = detail::tvec2<T>(m[0]);
        this->value[1] = detail::tvec2<T>(m[1]);
	}

    template <typename T> 
    inline tmat2x2<T>::tmat2x2(const tmat3x3<T>& m)
    {
        this->value[0] = detail::tvec2<T>(m[0]);
        this->value[1] = detail::tvec2<T>(m[1]);
    }

    template <typename T> 
    inline tmat2x2<T>::tmat2x2(const tmat4x4<T>& m)
    {
        this->value[0] = detail::tvec2<T>(m[0]);
        this->value[1] = detail::tvec2<T>(m[1]);
    }

	template <typename T> 
    inline tmat2x2<T>::tmat2x2(const tmat2x3<T>& m)
    {
        this->value[0] = detail::tvec2<T>(m[0]);
        this->value[1] = detail::tvec2<T>(m[1]);
    }

    template <typename T> 
    inline tmat2x2<T>::tmat2x2(const tmat3x2<T>& m)
    {
        this->value[0] = m[0];
        this->value[1] = m[1];
    }

    template <typename T> 
    inline tmat2x2<T>::tmat2x2(const tmat2x4<T>& m)
    {
        this->value[0] = detail::tvec2<T>(m[0]);
        this->value[1] = detail::tvec2<T>(m[1]);
    }

    template <typename T> 
    inline tmat2x2<T>::tmat2x2(const tmat4x2<T>& m)
    {
        this->value[0] = m[0];
        this->value[1] = m[1];
    }

    template <typename T> 
    inline tmat2x2<T>::tmat2x2(const tmat3x4<T>& m)
    {
        this->value[0] = detail::tvec2<T>(m[0]);
        this->value[1] = detail::tvec2<T>(m[1]);
    }

    template <typename T> 
    inline tmat2x2<T>::tmat2x2(const tmat4x3<T>& m)
    {
        this->value[0] = detail::tvec2<T>(m[0]);
        this->value[1] = detail::tvec2<T>(m[1]);
    }

/*
    template <typename T> 
    inline tmat2x2<T>::tmat2x2(const T* a)
    {
        this->value[0] = detail::tvec2<T>(a[0], a[1]);
        this->value[1] = detail::tvec2<T>(a[2], a[3]);
    }
*/

    template <typename T> 
    inline tmat2x2<T> tmat2x2<T>::_inverse() const
    {
        T Determinant = value[0][0] * value[1][1] - value[1][0] * value[0][1];

        tmat2x2<T> Inverse(
            + value[1][1] / Determinant,
            - value[1][0] / Determinant,
            - value[0][1] / Determinant, 
            + value[0][0] / Determinant);
        return Inverse;
    }

    //////////////////////////////////////////////////////////////
    // mat3 operators

    // This function shouldn't required but it seems that VC7.1 have an optimisation bug if this operator wasn't declared
    template <typename T> 
    inline tmat2x2<T>& tmat2x2<T>::operator=(tmat2x2<T> const & m)
    {
	    this->value[0] = m[0];
	    this->value[1] = m[1];
	    return *this;
    }

    template <typename T> 
    inline tmat2x2<T>& tmat2x2<T>::operator+= (const T & s)
    {
	    this->value[0] += s;
	    this->value[1] += s;
	    return *this;
    }

    template <typename T> 
    inline tmat2x2<T>& tmat2x2<T>::operator+= (tmat2x2<T> const & m)
    {
	    this->value[0] += m[0];
	    this->value[1] += m[1];
	    return *this;
    }

    template <typename T> 
    inline tmat2x2<T>& tmat2x2<T>::operator-= (const T & s)
    {
	    this->value[0] -= s;
	    this->value[1] -= s;
	    return *this;
    }

    template <typename T> 
    inline tmat2x2<T>& tmat2x2<T>::operator-= (tmat2x2<T> const & m)
    {
	    this->value[0] -= m[0];
	    this->value[1] -= m[1];
	    return *this;
    }

    template <typename T> 
    inline tmat2x2<T>& tmat2x2<T>::operator*= (const T & s)
    {
	    this->value[0] *= s;
	    this->value[1] *= s;
	    return *this;
    }

    template <typename T> 
    inline tmat2x2<T>& tmat2x2<T>::operator*= (tmat2x2<T> const & m)
    {
        return (*this = *this * m);
    }

    template <typename T> 
    inline tmat2x2<T>& tmat2x2<T>::operator/= (const T & s)
    {
	    this->value[0] /= s;
	    this->value[1] /= s;
	    return *this;
    }

    template <typename T> 
    inline tmat2x2<T>& tmat2x2<T>::operator/= (tmat2x2<T> const & m)
    {
        return (*this = *this / m);
    }

    template <typename T> 
    inline tmat2x2<T>& tmat2x2<T>::operator++ ()
    {
	    ++this->value[0];
	    ++this->value[1];
	    return *this;
    }

    template <typename T> 
    inline tmat2x2<T>& tmat2x2<T>::operator-- ()
    {
	    --this->value[0];
	    --this->value[1];
	    return *this;
    }

    //////////////////////////////////////////////////////////////
	// Binary operators

    template <typename T> 
    inline tmat2x2<T> operator+ (tmat2x2<T> const & m, const T & s)
    {
        return tmat2x2<T>(
            m[0] + s,
            m[1] + s);
    }

    template <typename T> 
    inline tmat2x2<T> operator+ (const T & s, tmat2x2<T> const & m)
    {
        return tmat2x2<T>(
            m[0] + s,
            m[1] + s);
    }

    //template <typename T> 
    //inline detail::tvec2<T> operator+ (tmat2x2<T> const & m, const detail::tvec2<T>& v)
    //{

    //}

    //template <typename T> 
    //inline detail::tvec2<T> operator+ (const detail::tvec2<T>& v, tmat2x2<T> const & m)
    //{

    //}

    template <typename T> 
    inline tmat2x2<T> operator+ (tmat2x2<T> const & m1, tmat2x2<T> const & m2)
    {
        return tmat2x2<T>(
            m1[0] + m2[0],
            m1[1] + m2[1]);
    }

    template <typename T> 
    inline tmat2x2<T> operator- (tmat2x2<T> const & m, const T & s)
    {
        return tmat2x2<T>(
            m[0] - s,
            m[1] - s);
    }

    template <typename T> 
    inline tmat2x2<T> operator- (const T & s, tmat2x2<T> const & m)
    {
        return tmat2x2<T>(
            s - m[0],
            s - m[1]);
    }

    //template <typename T> 
    //inline tmat2x2<T> operator- (tmat2x2<T> const & m, const detail::tvec2<T>& v)
    //{

    //}

    //template <typename T> 
    //inline tmat2x2<T> operator- (const detail::tvec2<T>& v, tmat2x2<T> const & m)
    //{

    //}

    template <typename T> 
    inline tmat2x2<T> operator- (tmat2x2<T> const & m1, tmat2x2<T> const & m2)
    {
        return tmat2x2<T>(
            m1[0] - m2[0],
            m1[1] - m2[1]);
    }

    template <typename T> 
    inline tmat2x2<T> operator* (tmat2x2<T> const & m, const T & s)
    {
        return tmat2x2<T>(
            m[0] * s,
            m[1] * s);
    }

    template <typename T> 
    inline tmat2x2<T> operator* (const T & s, tmat2x2<T> const & m)
    {
        return tmat2x2<T>(
            m[0] * s,
            m[1] * s);
    }

    template <typename T> 
    inline detail::tvec2<T> operator* (tmat2x2<T> const & m, const detail::tvec2<T>& v)
    {
        return detail::tvec2<T>(
            m[0][0] * v.x + m[1][0] * v.y,
            m[0][1] * v.x + m[1][1] * v.y);
    }

    template <typename T> 
    inline detail::tvec2<T> operator* (const detail::tvec2<T>& v, tmat2x2<T> const & m)
    {
        return detail::tvec2<T>(
            m[0][0] * v.x + m[0][1] * v.y,
            m[1][0] * v.x + m[1][1] * v.y);
    }

	template <typename T>
	inline tmat2x2<T> operator* (tmat2x2<T> const & m1, tmat2x2<T> const & m2)
	{
		return tmat2x2<T>(
			m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1],
			m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1],
			m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1],
			m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1]);
	}

    template <typename T> 
    inline tmat2x2<T> operator/ (tmat2x2<T> const & m, const T & s)
    {
        return tmat2x2<T>(
            m[0] / s,
            m[1] / s);
    }

    template <typename T> 
    inline tmat2x2<T> operator/ (const T & s, tmat2x2<T> const & m)
    {
        return tmat2x2<T>(
            s / m[0],
            s / m[1]);
    }

    template <typename T> 
    inline detail::tvec2<T> operator/ (tmat2x2<T> const & m, const detail::tvec2<T>& v)
    {
        return m._inverse() * v;
    }

    template <typename T> 
    inline detail::tvec2<T> operator/ (const detail::tvec2<T>& v, tmat2x2<T> const & m)
    {
        return v * m._inverse();
    }

    template <typename T> 
    inline tmat2x2<T> operator/ (tmat2x2<T> const & m1, tmat2x2<T> const & m2)
    {
        return m1 * m2._inverse();
    }

	// Unary constant operators
    template <typename valType> 
    inline tmat2x2<valType> const operator- 
	(
		tmat2x2<valType> const & m
	)
    {
        return tmat2x2<valType>(
            -m[0], 
            -m[1]);
    }

    template <typename valType> 
    inline tmat2x2<valType> const operator++ 
	(
		tmat2x2<valType> const & m, 
		int
	) 
    {
        return tmat2x2<valType>(
            m[0] + valType(1),
            m[1] + valType(1));
    }

    template <typename valType> 
    inline tmat2x2<valType> const operator-- 
	(
		tmat2x2<valType> const & m, 
		int
	) 
    {
        return tmat2x2<valType>(
            m[0] - valType(1),
            m[1] - valType(1));
    }

} //namespace detail
} //namespace glm
