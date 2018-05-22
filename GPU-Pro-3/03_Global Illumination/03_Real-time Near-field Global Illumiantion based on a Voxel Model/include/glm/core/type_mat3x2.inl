///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2006-08-05
// Updated : 2006-10-01
// Licence : This source is under MIT License
// File    : glm/core/type_mat3x2.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail{

	template <typename valType> 
	typename tmat3x2<valType>::size_type tmat3x2<valType>::col_size()
	{
		return typename tmat3x2<valType>::size_type(3);
	}

	template <typename valType> 
	typename tmat3x2<valType>::size_type tmat3x2<valType>::row_size()
	{
		return typename tmat3x2<valType>::size_type(2);
	}

	template <typename valType> 
	bool tmat3x2<valType>::is_matrix()
	{
		return true;
	}

	//////////////////////////////////////
	// Accesses

	template <typename valType>
	detail::tvec2<valType>& tmat3x2<valType>::operator[](typename tmat3x2<valType>::size_type i)
	{
		assert(
			i >= typename tmat3x2<valType>::size_type(0) && 
			i < tmat3x2<valType>::col_size());

		return value[i];
	}

	template <typename valType>
	const detail::tvec2<valType>& tmat3x2<valType>::operator[](typename tmat3x2<valType>::size_type i) const
	{
		assert(
			i >= typename tmat3x2<valType>::size_type(0) && 
			i < tmat3x2<valType>::col_size());

		return value[i];
	}

    //////////////////////////////////////////////////////////////
    // Constructors

    template <typename T> 
    inline tmat3x2<T>::tmat3x2()
    {
        this->value[0] = detail::tvec2<T>(1, 0);
        this->value[1] = detail::tvec2<T>(0, 1);
        this->value[2] = detail::tvec2<T>(0, 0);
    }

    template <typename T> 
    inline tmat3x2<T>::tmat3x2(const T f)
    {
        this->value[0] = detail::tvec2<T>(f, 0);
        this->value[1] = detail::tvec2<T>(0, f);
        this->value[2] = detail::tvec2<T>(0, 0);
    }

    template <typename T> 
    inline tmat3x2<T>::tmat3x2
    (
        const T x0, const T y0,
        const T x1, const T y1,
        const T x2, const T y2
    )
    {
        this->value[0] = detail::tvec2<T>(x0, y0);
        this->value[1] = detail::tvec2<T>(x1, y1);
        this->value[2] = detail::tvec2<T>(x2, y2);
    }

    template <typename T> 
    inline tmat3x2<T>::tmat3x2
    (
        const detail::tvec2<T> & v0, 
        const detail::tvec2<T> & v1, 
        const detail::tvec2<T> & v2
    )
    {
        this->value[0] = v0;
        this->value[1] = v1;
        this->value[2] = v2;
    }

    // Conversion
    template <typename T> 
    template <typename U> 
    inline tmat3x2<T>::tmat3x2(const tmat3x2<U>& m)
    {
        this->value[0] = detail::tvec2<T>(m[0]);
        this->value[1] = detail::tvec2<T>(m[1]);
        this->value[2] = detail::tvec2<T>(m[2]);
	}

    template <typename T> 
    inline tmat3x2<T>::tmat3x2(tmat2x2<T> const & m)
    {
        this->value[0] = m[0];
        this->value[1] = m[1];
        this->value[2] = detail::tvec2<T>(T(0));
    }

    template <typename T> 
    inline tmat3x2<T>::tmat3x2(const tmat3x3<T>& m)
    {
        this->value[0] = detail::tvec2<T>(m[0]);
        this->value[1] = detail::tvec2<T>(m[1]);
        this->value[2] = detail::tvec2<T>(m[2]);
    }

    template <typename T> 
    inline tmat3x2<T>::tmat3x2(const tmat4x4<T>& m)
    {
        this->value[0] = detail::tvec2<T>(m[0]);
        this->value[1] = detail::tvec2<T>(m[1]);
        this->value[2] = detail::tvec2<T>(m[2]);
    }

    template <typename T> 
    inline tmat3x2<T>::tmat3x2(const tmat2x3<T>& m)
    {
        this->value[0] = detail::tvec2<T>(m[0]);
        this->value[1] = detail::tvec2<T>(m[1]);
        this->value[2] = detail::tvec2<T>(T(0));
    }

    template <typename T> 
    inline tmat3x2<T>::tmat3x2(const tmat2x4<T>& m)
    {
        this->value[0] = detail::tvec2<T>(m[0]);
        this->value[1] = detail::tvec2<T>(m[1]);
        this->value[2] = detail::tvec2<T>(T(0));
    }

    template <typename T> 
    inline tmat3x2<T>::tmat3x2(const tmat3x4<T>& m)
    {
        this->value[0] = detail::tvec2<T>(m[0]);
        this->value[1] = detail::tvec2<T>(m[1]);
        this->value[2] = detail::tvec2<T>(m[2]);
    }

    template <typename T> 
    inline tmat3x2<T>::tmat3x2(const tmat4x2<T>& m)
    {
        this->value[0] = m[0];
        this->value[1] = m[1];
        this->value[2] = m[2];
    }

    template <typename T> 
    inline tmat3x2<T>::tmat3x2(const tmat4x3<T>& m)
    {
        this->value[0] = detail::tvec2<T>(m[0]);
        this->value[1] = detail::tvec2<T>(m[1]);
        this->value[2] = detail::tvec2<T>(m[2]);
    }

    //////////////////////////////////////////////////////////////
    // Unary updatable operators

    template <typename T> 
    inline tmat3x2<T>& tmat3x2<T>::operator= (const tmat3x2<T>& m)
    {
        this->value[0] = m[0];
        this->value[1] = m[1];
        this->value[2] = m[2];
        return *this;
    }

    template <typename T> 
    inline tmat3x2<T>& tmat3x2<T>::operator+= (const T & s)
    {
        this->value[0] += s;
        this->value[1] += s;
        this->value[2] += s;
        return *this;
    }

    template <typename T> 
    inline tmat3x2<T>& tmat3x2<T>::operator+= (const tmat3x2<T>& m)
    {
        this->value[0] += m[0];
        this->value[1] += m[1];
        this->value[2] += m[2];
        return *this;
    }

    template <typename T> 
    inline tmat3x2<T>& tmat3x2<T>::operator-= (const T & s)
    {
        this->value[0] -= s;
        this->value[1] -= s;
        this->value[2] -= s;
        return *this;
    }

    template <typename T> 
    inline tmat3x2<T>& tmat3x2<T>::operator-= (const tmat3x2<T>& m)
    {
        this->value[0] -= m[0];
        this->value[1] -= m[1];
        this->value[2] -= m[2];
        return *this;
    }

    template <typename T> 
    inline tmat3x2<T>& tmat3x2<T>::operator*= (const T & s)
    {
        this->value[0] *= s;
        this->value[1] *= s;
        this->value[2] *= s;
        return *this;
    }

    template <typename T> 
    inline tmat3x2<T>& tmat3x2<T>::operator*= (const tmat2x3<T>& m)
    {
        return (*this = tmat3x2<T>(*this * m));
    }

    template <typename T> 
    inline tmat3x2<T> & tmat3x2<T>::operator/= (const T & s)
    {
        this->value[0] /= s;
        this->value[1] /= s;
        this->value[2] /= s;
        return *this;
    }

    //template <typename T> 
    //inline tmat3x2<T>& tmat3x2<T>::operator/= (const tmat3x2<T>& m)
    //{
    //    return (*this = tmat3x2<T>(*this / m));
    //}

    template <typename T> 
    inline tmat3x2<T>& tmat3x2<T>::operator++ ()
    {
        ++this->value[0];
        ++this->value[1];
        ++this->value[2];
        return *this;
    }

    template <typename T> 
    inline tmat3x2<T>& tmat3x2<T>::operator-- ()
    {
        --this->value[0];
        --this->value[1];
        --this->value[2];
        return *this;
    }

    //////////////////////////////////////////////////////////////
    // inverse
    template <typename valType> 
    inline tmat2x3<valType> tmat3x2<valType>::_inverse() const
    {
		assert(0); //g.truc.creation[at]gmail.com
    }

    //////////////////////////////////////////////////////////////
    // Binary operators

    template <typename T> 
    inline tmat3x2<T> operator+ (const tmat3x2<T>& m, const T & s)
    {
        return tmat3x2<T>(
            m[0] + s,
            m[1] + s,
            m[2] + s);
    }

    template <typename T> 
    inline tmat3x2<T> operator+ (const tmat3x2<T>& m1, const tmat3x2<T>& m2)
    {
        return tmat3x2<T>(
            m1[0] + m2[0],
            m1[1] + m2[1],
            m1[2] + m2[2]);
    }

    template <typename T> 
    inline tmat3x2<T> operator- (const tmat3x2<T>& m, const T & s)
    {
        return tmat3x4<T>(
            m[0] - s,
            m[1] - s,
            m[2] - s);
    }

    template <typename T> 
    inline tmat3x2<T> operator- (const tmat3x2<T>& m1, const tmat3x2<T>& m2)
    {
        return tmat3x2<T>(
            m1[0] - m2[0],
            m1[1] - m2[1],
            m1[2] - m2[2]);
    }

    template <typename T> 
    inline tmat3x2<T> operator* (const tmat3x2<T>& m, const T & s)
    {
        return tmat3x2<T>(
            m[0] * s,
            m[1] * s,
            m[2] * s);
    }

    template <typename T> 
    inline tmat3x2<T> operator* (const T & s, const tmat3x2<T> & m)
    {
        return tmat3x2<T>(
            m[0] * s,
            m[1] * s,
            m[2] * s);
    }
   
    template <typename T>
    inline detail::tvec2<T> operator* (const tmat3x2<T>& m, const detail::tvec3<T>& v)
    {
        return detail::tvec2<T>(
            m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z,
            m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z);
    }

    template <typename T> 
    inline detail::tvec3<T> operator* (const detail::tvec2<T>& v, const tmat3x2<T>& m) 
    {
        return detail::tvec3<T>(
            m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w,
            m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w);
    }

    template <typename T> 
    inline tmat2x2<T> operator* (const tmat3x2<T>& m1, const tmat2x3<T>& m2)
    {
        const T SrcA00 = m1[0][0];
        const T SrcA01 = m1[0][1];
        const T SrcA10 = m1[1][0];
        const T SrcA11 = m1[1][1];
        const T SrcA20 = m1[2][0];
        const T SrcA21 = m1[2][1];

        const T SrcB00 = m2[0][0];
        const T SrcB01 = m2[0][1];
        const T SrcB02 = m2[0][2];
        const T SrcB10 = m2[1][0];
        const T SrcB11 = m2[1][1];
        const T SrcB12 = m2[1][2];

        tmat2x2<T> Result;
        Result[0][0] = SrcA00 * SrcB00 + SrcA01 * SrcB01 + SrcA20 * SrcB02;
        Result[0][1] = SrcA01 * SrcB00 + SrcA11 * SrcB01 + SrcA21 * SrcB02;
        Result[1][0] = SrcA00 * SrcB10 + SrcA10 * SrcB11 + SrcA20 * SrcB12;
        Result[1][1] = SrcA01 * SrcB10 + SrcA11 * SrcB11 + SrcA21 * SrcB12;
        return Result;
    }

    template <typename T> 
    inline tmat3x2<T> operator/ (const tmat3x2<T>& m, const T & s)
    {
        return tmat3x2<T>(
            m[0] / s,
            m[1] / s,
            m[2] / s,
            m[3] / s);        
    }

    template <typename T> 
    inline tmat3x2<T> operator/ (const T & s, const tmat3x2<T>& m)
    {
        return tmat3x2<T>(
            s / m[0],
            s / m[1],
            s / m[2],
            s / m[3]);        
    }

	//template <typename valType> 
	//inline tvec2<valType> operator/ 
	//(
	//	tmat3x2<valType> const & m, 
	//	tvec2<valType> const & v
	//)
	//{
	//	return m._inverse() * v;
	//}

	//template <typename valType> 
	//inline tvec3<valType> operator/ 
	//(
	//	tvec2<valType> const & v, 
	//	tmat3x2<valType> const & m
	//)
	//{
	//	return v * m._inverse();
	//}

	//template <typename valType> 
	//inline tmat3x3<valType> operator/ 
	//(
	//	tmat3x2<valType> const & m1, 
	//	tmat2x3<valType> const & m2
	//)
	//{
	//	return m1 * m2._inverse();
	//}

	// Unary constant operators
    template <typename valType> 
    inline tmat3x2<valType> const operator- 
	(
		tmat3x2<valType> const & m
	)
    {
        return tmat3x2<valType>(
            -m[0], 
            -m[1],
            -m[2]);
    }

    template <typename valType> 
    inline tmat3x2<valType> const operator++ 
	(
		tmat3x2<valType> const & m, 
		int
	) 
    {
        return tmat3x2<valType>(
            m[0] + valType(1),
            m[1] + valType(1),
            m[2] + valType(1));
    }

    template <typename valType> 
    inline tmat3x2<valType> const operator-- 
	(
		tmat3x2<valType> const & m, 
		int
	) 
    {
        return tmat3x2<valType>(
            m[0] - valType(1),
            m[1] - valType(1),
            m[2] - valType(1));
    }

} //namespace detail
} //namespace glm
