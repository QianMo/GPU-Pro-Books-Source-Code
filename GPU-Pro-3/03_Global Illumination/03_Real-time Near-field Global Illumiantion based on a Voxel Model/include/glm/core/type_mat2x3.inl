///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2006-08-05
// Updated : 2006-10-01
// Licence : This source is under MIT License
// File    : glm/core/type_mat2x3.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail{

	template <typename valType> 
	typename tmat2x3<valType>::size_type tmat2x3<valType>::col_size()
	{
		return typename tmat2x3<valType>::size_type(2);
	}

	template <typename valType> 
	typename tmat2x3<valType>::size_type tmat2x3<valType>::row_size()
	{
		return typename tmat2x3<valType>::size_type(3);
	}

	template <typename valType> 
	bool tmat2x3<valType>::is_matrix()
	{
		return true;
	}

	//////////////////////////////////////
	// Accesses

	template <typename valType>
	detail::tvec3<valType>& tmat2x3<valType>::operator[](typename tmat2x3<valType>::size_type i)
	{
		assert(
			i >= typename tmat2x3<valType>::size_type(0) && 
			i < tmat2x3<valType>::col_size());

		return value[i];
	}

	template <typename valType>
	const detail::tvec3<valType>& tmat2x3<valType>::operator[](typename tmat2x3<valType>::size_type i) const
	{
		assert(
			i >= typename tmat2x3<valType>::size_type(0) && 
			i < tmat2x3<valType>::col_size());

		return value[i];
	}

    //////////////////////////////////////////////////////////////
    // Constructors

    template <typename T> 
    inline tmat2x3<T>::tmat2x3()
    {
        this->value[0] = detail::tvec3<T>(T(1), T(0), T(0));
        this->value[1] = detail::tvec3<T>(T(0), T(1), T(0));
    }

    template <typename T> 
    inline tmat2x3<T>::tmat2x3(const T f)
    {
        this->value[0] = detail::tvec3<T>(f, T(0), T(0));
        this->value[1] = detail::tvec3<T>(T(0), f, T(0));
    }

    template <typename T> 
    inline tmat2x3<T>::tmat2x3
    (
        const T x0, const T y0, const T z0,
        const T x1, const T y1, const T z1
    )
    {
        this->value[0] = detail::tvec3<T>(x0, y0, z0);
        this->value[1] = detail::tvec3<T>(x1, y1, z1);
    }

    template <typename T> 
    inline tmat2x3<T>::tmat2x3
    (
        const detail::tvec3<T> & v0, 
        const detail::tvec3<T> & v1
    )
    {
        this->value[0] = v0;
        this->value[1] = v1;
    }

    // Conversion
    template <typename T> 
    template <typename U> 
    inline tmat2x3<T>::tmat2x3(const tmat2x3<U>& m)
    {
        this->value[0] = detail::tvec3<T>(m[0]);
        this->value[1] = detail::tvec3<T>(m[1]);
	}

    template <typename T> 
    inline tmat2x3<T>::tmat2x3(tmat2x2<T> const & m)
    {
        this->value[0] = detail::tvec3<T>(m[0], T(0));
        this->value[1] = detail::tvec3<T>(m[1], T(0));
    }

    template <typename T> 
    inline tmat2x3<T>::tmat2x3(const tmat3x3<T>& m)
    {
        this->value[0] = detail::tvec3<T>(m[0]);
        this->value[1] = detail::tvec3<T>(m[1]);
    }

    template <typename T> 
    inline tmat2x3<T>::tmat2x3(const tmat4x4<T>& m)
    {
        this->value[0] = detail::tvec3<T>(m[0]);
        this->value[1] = detail::tvec3<T>(m[1]);
    }

    template <typename T> 
    inline tmat2x3<T>::tmat2x3(const tmat2x4<T>& m)
    {
        this->value[0] = detail::tvec3<T>(m[0]);
        this->value[1] = detail::tvec3<T>(m[1]);
    }

    template <typename T> 
    inline tmat2x3<T>::tmat2x3(const tmat3x2<T>& m)
    {
        this->value[0] = detail::tvec3<T>(m[0], T(0));
        this->value[1] = detail::tvec3<T>(m[1], T(0));
    }

    template <typename T> 
    inline tmat2x3<T>::tmat2x3(const tmat3x4<T>& m)
    {
        this->value[0] = detail::tvec3<T>(m[0]);
        this->value[1] = detail::tvec3<T>(m[1]);
    }

    template <typename T> 
    inline tmat2x3<T>::tmat2x3(const tmat4x2<T>& m)
    {
        this->value[0] = detail::tvec3<T>(m[0], T(0));
        this->value[1] = detail::tvec3<T>(m[1], T(0));
    }

    template <typename T> 
    inline tmat2x3<T>::tmat2x3(const tmat4x3<T>& m)
    {
        this->value[0] = m[0];
        this->value[1] = m[1];
    }

    //////////////////////////////////////////////////////////////
    // Unary updatable operators

    template <typename T> 
    inline tmat2x3<T>& tmat2x3<T>::operator= (const tmat2x3<T>& m)
    {
        this->value[0] = m[0];
        this->value[1] = m[1];
        return *this;
    }

    template <typename T> 
    inline tmat2x3<T>& tmat2x3<T>::operator+= (const T & s)
    {
        this->value[0] += s;
        this->value[1] += s;
        return *this;
    }

    template <typename T> 
    inline tmat2x3<T>& tmat2x3<T>::operator+= (const tmat2x3<T>& m)
    {
        this->value[0] += m[0];
        this->value[1] += m[1];
        return *this;
    }

    template <typename T> 
    inline tmat2x3<T>& tmat2x3<T>::operator-= (const T & s)
    {
        this->value[0] -= s;
        this->value[1] -= s;
        return *this;
    }

    template <typename T> 
    inline tmat2x3<T>& tmat2x3<T>::operator-= (const tmat2x3<T>& m)
    {
        this->value[0] -= m[0];
        this->value[1] -= m[1];
        return *this;
    }

    template <typename T> 
    inline tmat2x3<T>& tmat2x3<T>::operator*= (const T & s)
    {
        this->value[0] *= s;
        this->value[1] *= s;
        return *this;
    }

    template <typename T> 
    inline tmat2x3<T>& tmat2x3<T>::operator*= (const tmat3x2<T>& m)
    {
        return (*this = tmat2x3<T>(*this * m));
    }

    template <typename T> 
    inline tmat2x3<T> & tmat2x3<T>::operator/= (const T & s)
    {
        this->value[0] /= s;
        this->value[1] /= s;
        return *this;
    }
/* ToDo
    template <typename T> 
    inline tmat2x3<T>& tmat2x3<T>::operator/= (const tmat3x2<T>& m)
    {
        return (*this = tmat2x3<T>(*this / m));
    }
*/
    template <typename T> 
    inline tmat2x3<T>& tmat2x3<T>::operator++ ()
    {
        ++this->value[0];
        ++this->value[1];
        return *this;
    }

    template <typename T> 
    inline tmat2x3<T>& tmat2x3<T>::operator-- ()
    {
        --this->value[0];
        --this->value[1];
        return *this;
    }
    
    //////////////////////////////////////////////////////////////
    // Unary constant operators
    template <typename T> 
    inline const tmat2x3<T> tmat2x3<T>::operator- () const
    {
        return tmat2x3<T>(
            -this->value[0], 
            -this->value[1]);
    }

    template <typename T> 
    inline const tmat2x3<T> tmat2x3<T>::operator-- (int n) const 
    {
        tmat2x3<T> m = *this;
        --m[0];
        --m[1];
        return m;
    }

    template <typename T> 
    inline const tmat2x3<T> tmat2x3<T>::operator++ (int n) const
    {
        tmat2x3<T> m = *this;
        ++m[0];
        ++m[1];
        return m;
    }

    //////////////////////////////////////////////////////////////
    // Binary operators

    template <typename T> 
    inline tmat2x3<T> operator+ (const tmat2x3<T>& m, const T & s)
    {
        return tmat2x3<T>(
            m[0] + s,
            m[1] + s);
    }

    template <typename T> 
    inline tmat2x3<T> operator+ (const tmat2x3<T>& m1, const tmat2x3<T>& m2)
    {
        return tmat2x3<T>(
            m1[0] + m2[0],
            m1[1] + m2[1]);
    }

    template <typename T> 
    inline tmat2x3<T> operator- (const tmat2x3<T>& m, const T & s)
    {
        return tmat2x3<T>(
            m[0] - s,
            m[1] - s);
    }

    template <typename T> 
    inline tmat2x3<T> operator- (const tmat2x3<T>& m1, const tmat2x3<T>& m2)
    {
        return tmat2x3<T>(
            m1[0] - m2[0],
            m1[1] - m2[1]);
    }

    template <typename T> 
    inline tmat2x3<T> operator* (const tmat2x3<T>& m, const T & s)
    {
        return tmat2x3<T>(
            m[0] * s,
            m[1] * s);
    }

    template <typename T> 
    inline tmat2x3<T> operator* (const T & s, const tmat2x3<T> & m)
    {
        return tmat2x3<T>(
            m[0] * s,
            m[1] * s);
    }
   
    template <typename T>
    inline detail::tvec3<T> operator* (const tmat2x3<T>& m, const detail::tvec2<T>& v)
    {
        return detail::tvec3<T>(
            m[0][0] * v.x + m[1][0] * v.y,
            m[0][1] * v.x + m[1][1] * v.y,
            m[0][2] * v.x + m[1][2] * v.y,
            m[0][3] * v.x + m[1][3] * v.y);
    }

    template <typename T> 
    inline detail::tvec2<T> operator* (const detail::tvec3<T>& v, const tmat2x3<T>& m) 
    {
        return detail::tvec2<T>(
            m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w,
            m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w);
    }

    template <typename T> 
    inline tmat3x3<T> operator* (const tmat2x3<T>& m1, const tmat3x2<T>& m2)
    {
        const T SrcA00 = m1[0][0];
        const T SrcA01 = m1[0][1];
        const T SrcA02 = m1[0][2];
        const T SrcA10 = m1[1][0];
        const T SrcA11 = m1[1][1];
        const T SrcA12 = m1[1][2];

        const T SrcB00 = m2[0][0];
        const T SrcB01 = m2[0][1];
        const T SrcB10 = m2[1][0];
        const T SrcB11 = m2[1][1];
        const T SrcB20 = m2[2][0];
        const T SrcB21 = m2[2][1];

        tmat3x3<T> Result;
        Result[0][0] = SrcA00 * SrcB00 + SrcA10 * SrcB01;
        Result[0][1] = SrcA01 * SrcB00 + SrcA11 * SrcB01;
        Result[0][2] = SrcA02 * SrcB00 + SrcA12 * SrcB01;
        Result[1][0] = SrcA00 * SrcB10 + SrcA10 * SrcB11;
        Result[1][1] = SrcA01 * SrcB10 + SrcA11 * SrcB11;
        Result[1][2] = SrcA02 * SrcB10 + SrcA12 * SrcB11;
        Result[2][0] = SrcA00 * SrcB20 + SrcA10 * SrcB21;
        Result[2][1] = SrcA01 * SrcB20 + SrcA11 * SrcB21;
        Result[2][2] = SrcA02 * SrcB20 + SrcA12 * SrcB21;
        return Result;
    }

    template <typename T> 
    inline tmat2x3<T> operator/ (const tmat2x3<T>& m, const T & s)
    {
        return tmat2x3<T>(
            m[0] / s,
            m[1] / s,
            m[2] / s);        
    }

    template <typename T> 
    inline tmat2x3<T> operator/ (const T & s, const tmat2x3<T>& m)
    {
        return tmat2x3<T>(
            s / m[0],
            s / m[1],
            s / m[2]);        
    }

	//template <typename valType> 
	//tvec3<valType> operator/ 
	//(
	//	tmat3x2<valType> const & m, 
	//	tvec2<valType> const & v
	//)
	//{
	//	return m._inverse() * v;
	//}

	//template <typename valType> 
	//tvec3<valType> operator/ 
	//(
	//	tvec2<valType> const & v, 
	//	tmat3x2<valType> const & m
	//)
	//{
	//	return v * m._inverse();
	//}

	//template <typename valType> 
	//inline tmat2x2<valType> operator/ 
	//(
	//	tmat3x2<valType> const & m1, 
	//	tmat2x3<valType> const & m2
	//)
	//{
	//	return m1 * m2._inverse();
	//}

	// Unary constant operators
    template <typename valType> 
    inline tmat2x3<valType> const operator- 
	(
		tmat2x3<valType> const & m
	)
    {
        return tmat2x3<valType>(
            -m[0], 
            -m[1]);
    }

    template <typename valType> 
    inline tmat2x3<valType> const operator++ 
	(
		tmat2x3<valType> const & m, 
		int
	) 
    {
        return tmat2x3<valType>(
            m[0] + valType(1),
            m[1] + valType(1));
    }

    template <typename valType> 
    inline tmat2x3<valType> const operator-- 
	(
		tmat2x3<valType> const & m, 
		int
	) 
    {
        return tmat2x3<valType>(
            m[0] - valType(1),
            m[1] - valType(1));
    }

} //namespace detail
} //namespace glm
