///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2006-04-17
// Updated : 2006-04-17
// Licence : This source is under MIT License
// File    : glm/core/type_mat4x3.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail{

	template <typename valType> 
	typename tmat4x3<valType>::size_type tmat4x3<valType>::col_size()
	{
		return typename tmat4x3<valType>::size_type(4);
	}

	template <typename valType> 
	typename tmat4x3<valType>::size_type tmat4x3<valType>::row_size()
	{
		return typename tmat4x3<valType>::size_type(3);
	}

	template <typename valType> 
	bool tmat4x3<valType>::is_matrix()
	{
		return true;
	}

	//////////////////////////////////////
	// Accesses

	template <typename valType>
	detail::tvec3<valType>& tmat4x3<valType>::operator[]
	(
		typename tmat4x3<valType>::size_type i
	)
	{
		assert(
			i >= typename tmat4x3<valType>::size_type(0) && 
			i < tmat4x3<valType>::col_size());

		return value[i];
	}

	template <typename valType>
	const detail::tvec3<valType>& tmat4x3<valType>::operator[]
	(
		typename tmat4x3<valType>::size_type i
	) const
	{
		assert(
			i >= typename tmat4x3<valType>::size_type(0) && 
			i < tmat4x3<valType>::col_size());

		return value[i];
	}

    //////////////////////////////////////////////////////////////
    // Constructors

    template <typename valType> 
    inline tmat4x3<valType>::tmat4x3()
    {
        this->value[0] = detail::tvec3<valType>(1, 0, 0);
        this->value[1] = detail::tvec3<valType>(0, 1, 0);
        this->value[2] = detail::tvec3<valType>(0, 0, 1);
        this->value[3] = detail::tvec3<valType>(0, 0, 0);
    }

    template <typename valType> 
    inline tmat4x3<valType>::tmat4x3(valType const & s)
    {
        this->value[0] = detail::tvec3<valType>(s, 0, 0);
        this->value[1] = detail::tvec3<valType>(0, s, 0);
        this->value[2] = detail::tvec3<valType>(0, 0, s);
        this->value[3] = detail::tvec3<valType>(0, 0, 0);
    }

    template <typename valType> 
    inline tmat4x3<valType>::tmat4x3
    (
        const valType x0, const valType y0, const valType z0,
        const valType x1, const valType y1, const valType z1,
        const valType x2, const valType y2, const valType z2,
        const valType x3, const valType y3, const valType z3
    )
    {
        this->value[0] = detail::tvec3<valType>(x0, y0, z0);
        this->value[1] = detail::tvec3<valType>(x1, y1, z1);
        this->value[2] = detail::tvec3<valType>(x2, y2, z2);
        this->value[3] = detail::tvec3<valType>(x3, y3, z3);
    }

    template <typename valType> 
    inline tmat4x3<valType>::tmat4x3
    (
        const detail::tvec3<valType> & v0, 
        const detail::tvec3<valType> & v1, 
        const detail::tvec3<valType> & v2,
        const detail::tvec3<valType> & v3
    )
    {
        this->value[0] = v0;
        this->value[1] = v1;
        this->value[2] = v2;
        this->value[3] = v3;
    }

    // Conversion
    template <typename valType> 
    template <typename U> 
    inline tmat4x3<valType>::tmat4x3
	(
		tmat4x3<U> const & m
	)
    {
        this->value[0] = detail::tvec3<valType>(m[0]);
        this->value[1] = detail::tvec3<valType>(m[1]);
        this->value[2] = detail::tvec3<valType>(m[2]);
        this->value[3] = detail::tvec3<valType>(m[3]);
	}

    template <typename valType> 
    inline tmat4x3<valType>::tmat4x3
	(
		tmat2x2<valType> const & m
	)
    {
        this->value[0] = detail::tvec3<valType>(m[0], valType(0));
        this->value[1] = detail::tvec3<valType>(m[1], valType(0));
        this->value[2] = detail::tvec3<valType>(m[2], valType(1));
        this->value[3] = detail::tvec3<valType>(valType(0));
    }

    template <typename valType> 
    inline tmat4x3<valType>::tmat4x3
	(
		tmat3x3<valType> const & m
	)
    {
        this->value[0] = detail::tvec3<valType>(m[0]);
        this->value[1] = detail::tvec3<valType>(m[1]);
        this->value[2] = detail::tvec3<valType>(m[2]);
        this->value[3] = detail::tvec3<valType>(valType(0));
    }

    template <typename valType> 
    inline tmat4x3<valType>::tmat4x3
	(
		tmat4x4<valType> const & m
	)
    {
        this->value[0] = detail::tvec3<valType>(m[0]);
        this->value[1] = detail::tvec3<valType>(m[1]);
        this->value[2] = detail::tvec3<valType>(m[2]);
        this->value[3] = detail::tvec3<valType>(m[3]);
    }

    template <typename valType> 
    inline tmat4x3<valType>::tmat4x3
	(
		tmat2x3<valType> const & m
	)
    {
        this->value[0] = detail::tvec3<valType>(m[0]);
        this->value[1] = detail::tvec3<valType>(m[1]);
        this->value[2] = detail::tvec3<valType>(valType(0), valType(0), valType(1));
        this->value[3] = detail::tvec3<valType>(valType(0));
    }

    template <typename valType> 
    inline tmat4x3<valType>::tmat4x3
	(
		tmat3x2<valType> const & m
	)
    {
        this->value[0] = detail::tvec3<valType>(m[0], valType(0));
        this->value[1] = detail::tvec3<valType>(m[1], valType(0));
        this->value[2] = detail::tvec3<valType>(m[2], valType(1));
        this->value[3] = detail::tvec3<valType>(valType(0));
    }

    template <typename valType> 
    inline tmat4x3<valType>::tmat4x3
	(
		tmat2x4<valType> const & m
	)
    {
        this->value[0] = detail::tvec3<valType>(m[0]);
        this->value[1] = detail::tvec3<valType>(m[1]);
        this->value[2] = detail::tvec3<valType>(valType(0), valType(0), valType(1));
        this->value[3] = detail::tvec3<valType>(valType(0));
    }

    template <typename valType> 
    inline tmat4x3<valType>::tmat4x3
	(
		tmat4x2<valType> const & m
	)
    {
        this->value[0] = detail::tvec3<valType>(m[0], valType(0));
        this->value[1] = detail::tvec3<valType>(m[1], valType(0));
        this->value[2] = detail::tvec3<valType>(m[2], valType(1));
        this->value[3] = detail::tvec3<valType>(m[3], valType(0));
    }

    template <typename valType> 
    inline tmat4x3<valType>::tmat4x3
	(
		tmat3x4<valType> const & m
	)
    {
        this->value[0] = detail::tvec3<valType>(m[0]);
        this->value[1] = detail::tvec3<valType>(m[1]);
        this->value[2] = detail::tvec3<valType>(m[2]);
        this->value[3] = detail::tvec3<valType>(valType(0));
    }

    //////////////////////////////////////////////////////////////
    // Unary updatable operators

    template <typename valType> 
    inline tmat4x3<valType>& tmat4x3<valType>::operator= 
	(
		tmat4x3<valType> const & m
	)
    {
        this->value[0] = m[0];
        this->value[1] = m[1];
        this->value[2] = m[2];
        this->value[3] = m[3];
        return *this;
    }

    template <typename valType> 
    inline tmat4x3<valType>& tmat4x3<valType>::operator+= 
	(
		valType const & s
	)
    {
        this->value[0] += s;
        this->value[1] += s;
        this->value[2] += s;
        this->value[3] += s;
        return *this;
    }

    template <typename valType> 
    inline tmat4x3<valType>& tmat4x3<valType>::operator+= 
	(
		tmat4x3<valType> const & m
	)
    {
        this->value[0] += m[0];
        this->value[1] += m[1];
        this->value[2] += m[2];
        this->value[3] += m[3];
        return *this;
    }

    template <typename valType> 
    inline tmat4x3<valType>& tmat4x3<valType>::operator-= 
	(
		valType const & s
	)
    {
        this->value[0] -= s;
        this->value[1] -= s;
        this->value[2] -= s;
        this->value[3] -= s;
        return *this;
    }

    template <typename valType> 
    inline tmat4x3<valType>& tmat4x3<valType>::operator-= 
	(
		tmat4x3<valType> const & m
	)
    {
        this->value[0] -= m[0];
        this->value[1] -= m[1];
        this->value[2] -= m[2];
        this->value[3] -= m[3];
        return *this;
    }

    template <typename valType> 
    inline tmat4x3<valType>& tmat4x3<valType>::operator*= 
	(
		valType const & s
	)
    {
        this->value[0] *= s;
        this->value[1] *= s;
        this->value[2] *= s;
        this->value[3] *= s;
        return *this;
    }

    template <typename valType> 
    inline tmat4x3<valType>& tmat4x3<valType>::operator*= 
	(
		tmat3x4<valType> const & m
	)
    {
        return (*this = tmat4x3<valType>(*this * m));
    }

    template <typename valType> 
    inline tmat4x3<valType> & tmat4x3<valType>::operator/= 
	(
		valType const & s
	)
    {
        this->value[0] /= s;
        this->value[1] /= s;
        this->value[2] /= s;
        this->value[3] /= s;
        return *this;
    }

	//template <typename valType> 
	//inline tmat4x3<valType>& tmat4x3<valType>::operator/= 
	//(
	//	tmat3x4<valType> const & m
	//)
	//{
	//	return (*this = *this / m);
	//}

    template <typename valType> 
    inline tmat4x3<valType>& tmat4x3<valType>::operator++ ()
    {
        ++this->value[0];
        ++this->value[1];
        ++this->value[2];
        ++this->value[3];
        return *this;
    }

    template <typename valType> 
    inline tmat4x3<valType>& tmat4x3<valType>::operator-- ()
    {
        --this->value[0];
        --this->value[1];
        --this->value[2];
        --this->value[3];
        return *this;
    }

    //////////////////////////////////////////////////////////////
    // inverse
    template <typename valType> 
    inline tmat3x4<valType> tmat4x3<valType>::_inverse() const
    {
		assert(0); //g.truc.creation[at]gmail.com
    }

    //////////////////////////////////////////////////////////////
    // Unary constant operators
    template <typename valType> 
    inline const tmat4x3<valType> tmat4x3<valType>::operator- () const
    {
        return tmat4x3<valType>(
            -this->value[0], 
            -this->value[1],
            -this->value[2],
            -this->value[3]);
    }

    template <typename valType> 
    inline const tmat4x3<valType> tmat4x3<valType>::operator-- (int) const 
    {
        tmat4x3<valType> m = *this;
        --m.value[0];
        --m.value[1];
        --m.value[2];
        --m.value[3];
        return m;
    }

    template <typename valType> 
    inline const tmat4x3<valType> tmat4x3<valType>::operator++ (int) const
    {
        tmat4x4<valType> m = *this;
        ++m.value[0];
        ++m.value[1];
        ++m.value[2];
        ++m.value[3];
        return m;
    }

    //////////////////////////////////////////////////////////////
    // Binary operators

    template <typename valType> 
    inline tmat4x3<valType> operator+ (const tmat4x3<valType>& m, valType const & s)
    {
        return tmat4x3<valType>(
            m[0] + s,
            m[1] + s,
            m[2] + s,
            m[3] + s);
    }

    template <typename valType> 
    inline tmat4x3<valType> operator+ (const tmat4x3<valType>& m1, const tmat4x3<valType>& m2)
    {
        return tmat4x3<valType>(
            m1[0] + m2[0],
            m1[1] + m2[1],
            m1[2] + m2[2],
            m1[3] + m2[3]);
    }

    template <typename valType> 
    inline tmat4x3<valType> operator- (const tmat4x3<valType>& m, valType const & s)
    {
        return tmat4x3<valType>(
            m[0] - s,
            m[1] - s,
            m[2] - s,
            m[3] - s);
    }

    template <typename valType> 
    inline tmat4x3<valType> operator- (const tmat4x3<valType>& m1, const tmat4x3<valType>& m2)
    {
        return tmat4x3<valType>(
            m1[0] - m2[0],
            m1[1] - m2[1],
            m1[2] - m2[2],
            m1[3] - m2[3]);
    }

    template <typename valType> 
    inline tmat4x3<valType> operator* (const tmat4x3<valType>& m, valType const & s)
    {
        return tmat4x3<valType>(
            m[0] * s,
            m[1] * s,
            m[2] * s,
            m[3] * s);
    }

    template <typename valType> 
    inline tmat4x3<valType> operator* (valType const & s, const tmat4x3<valType> & m)
    {
        return tmat4x3<valType>(
            m[0] * s,
            m[1] * s,
            m[2] * s,
            m[3] * s);
    }
   
    template <typename valType>
    inline detail::tvec3<valType> operator* (const tmat4x3<valType>& m, const detail::tvec4<valType>& v)
    {
        return detail::tvec3<valType>(
            m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w,
            m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w,
            m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2] * v.w);
    }

    template <typename valType> 
    inline detail::tvec4<valType> operator* (const detail::tvec3<valType>& v, const tmat4x3<valType>& m) 
    {
        return detail::tvec4<valType>(
            v.x * m[0][0] + v.y * m[0][1] + v.z * m[0][2],
            v.x * m[1][0] + v.y * m[1][1] + v.z * m[1][2],
            v.x * m[2][0] + v.y * m[2][1] + v.z * m[2][2],
            v.x * m[3][0] + v.y * m[3][1] + v.z * m[3][2]);
    }

    template <typename valType> 
    inline tmat3x3<valType> operator* (const tmat4x3<valType>& m1, const tmat3x4<valType>& m2)
    {
        const valType SrcA00 = m1[0][0];
        const valType SrcA01 = m1[0][1];
        const valType SrcA02 = m1[0][2];
        const valType SrcA10 = m1[1][0];
        const valType SrcA11 = m1[1][1];
        const valType SrcA12 = m1[1][2];
        const valType SrcA20 = m1[2][0];
        const valType SrcA21 = m1[2][1];
        const valType SrcA22 = m1[2][2];
        const valType SrcA30 = m1[3][0];
        const valType SrcA31 = m1[3][1];
        const valType SrcA32 = m1[3][2];

        const valType SrcB00 = m2[0][0];
        const valType SrcB01 = m2[0][1];
        const valType SrcB02 = m2[0][2];
        const valType SrcB03 = m2[0][3];
        const valType SrcB10 = m2[1][0];
        const valType SrcB11 = m2[1][1];
        const valType SrcB12 = m2[1][2];
        const valType SrcB13 = m2[1][3];
        const valType SrcB20 = m2[2][0];
        const valType SrcB21 = m2[2][1];
        const valType SrcB22 = m2[2][2];
        const valType SrcB23 = m2[2][3];

        tmat3x3<valType> Result;
        Result[0][0] = SrcA00 * SrcB00 + SrcA10 * SrcB01 + SrcA20 * SrcB02 + SrcA30 * SrcB03;
        Result[0][1] = SrcA01 * SrcB00 + SrcA11 * SrcB01 + SrcA21 * SrcB02 + SrcA31 * SrcB03;
        Result[0][2] = SrcA02 * SrcB00 + SrcA12 * SrcB01 + SrcA22 * SrcB02 + SrcA32 * SrcB03;
        Result[1][0] = SrcA00 * SrcB10 + SrcA10 * SrcB11 + SrcA20 * SrcB12 + SrcA30 * SrcB13;
        Result[1][1] = SrcA01 * SrcB10 + SrcA11 * SrcB11 + SrcA21 * SrcB12 + SrcA31 * SrcB13;
        Result[1][2] = SrcA02 * SrcB10 + SrcA12 * SrcB11 + SrcA22 * SrcB12 + SrcA32 * SrcB13;
        Result[2][0] = SrcA00 * SrcB20 + SrcA10 * SrcB21 + SrcA20 * SrcB22 + SrcA30 * SrcB23;
        Result[2][1] = SrcA01 * SrcB20 + SrcA11 * SrcB21 + SrcA21 * SrcB22 + SrcA31 * SrcB23;
        Result[2][2] = SrcA02 * SrcB20 + SrcA12 * SrcB21 + SrcA22 * SrcB22 + SrcA32 * SrcB23;
        return Result;
    }

    template <typename valType> 
    inline tmat4x3<valType> operator/ (const tmat4x3<valType>& m, valType const & s)
    {
        return tmat4x3<valType>(
            m[0] / s,
            m[1] / s,
            m[2] / s,
            m[3] / s);        
    }

    template <typename valType> 
    inline tmat4x3<valType> operator/ (valType const & s, const tmat4x3<valType>& m)
    {
        return tmat4x3<valType>(
            s / m[0],
            s / m[1],
            s / m[2],
            s / m[3]);        
    }

	//template <typename valType> 
	//inline tvec3<valType> operator/ 
	//(
	//	tmat4x3<valType> const & m, 
	//	tvec3<valType> const & v
	//)
	//{
	//	return m._inverse() * v;
	//}

	//template <typename valType> 
	//inline tvec4<valType> operator/ 
	//(
	//	tvec3<valType> const & v, 
	//	tmat4x3<valType> const & m
	//)
	//{
	//	return v * m._inverse();
	//}

	//template <typename valType> 
	//inline tmat3x3<valType> operator/ 
	//(
	//	tmat4x3<valType> const & m1, 
	//	tmat3x4<valType> const & m2
	//)
	//{
	//	return m1 * m2._inverse();
	//}

	// Unary constant operators
    template <typename valType> 
    inline tmat4x3<valType> const operator- 
	(
		tmat4x3<valType> const & m
	)
    {
        return tmat4x3<valType>(
            -m[0], 
            -m[1],
            -m[2],
            -m[3]);
    }

    template <typename valType> 
    inline tmat4x3<valType> const operator++ 
	(
		tmat4x3<valType> const & m, 
		int
	) 
    {
        return tmat4x3<valType>(
            m[0] + valType(1),
            m[1] + valType(1),
            m[2] + valType(1),
            m[3] + valType(1));
    }

    template <typename valType> 
    inline tmat4x3<valType> const operator-- 
	(
		tmat4x3<valType> const & m, 
		int
	) 
    {
        return tmat4x3<valType>(
            m[0] - valType(1),
            m[1] - valType(1),
            m[2] - valType(1),
            m[3] - valType(1));
    }

} //namespace detail
} //namespace glm

