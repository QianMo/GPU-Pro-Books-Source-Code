///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2006-10-01
// Updated : 2008-10-05
// Licence : This source is under MIT License
// File    : glm/core/type_mat4x2.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail{

	template <typename valType> 
	typename tmat4x2<valType>::size_type tmat4x2<valType>::col_size()
	{
		return typename tmat4x2<valType>::size_type(4);
	}

	template <typename valType> 
	typename tmat4x2<valType>::size_type tmat4x2<valType>::row_size()
	{
		return typename tmat4x2<valType>::size_type(2);
	}

	template <typename valType> 
	bool tmat4x2<valType>::is_matrix()
	{
		return true;
	}

	//////////////////////////////////////
	// Accesses

	template <typename valType>
	detail::tvec2<valType>& tmat4x2<valType>::operator[]
	(
		typename tmat4x2<valType>::size_type i
	)
	{
		assert(
			i >= typename tmat4x2<valType>::size_type(0) && 
			i < tmat4x2<valType>::col_size());

		return value[i];
	}

	template <typename valType>
	const detail::tvec2<valType>& tmat4x2<valType>::operator[]
	(
		typename tmat4x2<valType>::size_type i
	) const
	{
		assert(
			i >= typename tmat4x2<valType>::size_type(0) && 
			i < tmat4x2<valType>::col_size());

		return value[i];
	}

    //////////////////////////////////////////////////////////////
    // Constructors

    template <typename valType> 
    inline tmat4x2<valType>::tmat4x2()
    {
        this->value[0] = detail::tvec2<valType>(1, 0);
        this->value[1] = detail::tvec2<valType>(0, 1);
        this->value[2] = detail::tvec2<valType>(0, 0);
        this->value[3] = detail::tvec2<valType>(0, 0);
    }

    template <typename valType> 
    inline tmat4x2<valType>::tmat4x2
	(
		valType const & s
	)
    {
        this->value[0] = detail::tvec2<valType>(s, 0);
        this->value[1] = detail::tvec2<valType>(0, s);
        this->value[2] = detail::tvec2<valType>(0, 0);
        this->value[3] = detail::tvec2<valType>(0, 0);
    }

    template <typename valType> 
    inline tmat4x2<valType>::tmat4x2
    (
        valType const & x0, valType const & y0,
        valType const & x1, valType const & y1,
        valType const & x2, valType const & y2,
        valType const & x3, valType const & y3
    )
    {
        this->value[0] = detail::tvec2<valType>(x0, y0);
        this->value[1] = detail::tvec2<valType>(x1, y1);
        this->value[2] = detail::tvec2<valType>(x2, y2);
        this->value[3] = detail::tvec2<valType>(x3, y3);
    }

    template <typename valType> 
    inline tmat4x2<valType>::tmat4x2
    (
        const detail::tvec2<valType> & v0, 
        const detail::tvec2<valType> & v1, 
        const detail::tvec2<valType> & v2,
        const detail::tvec2<valType> & v3
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
    inline tmat4x2<valType>::tmat4x2(const tmat4x2<U>& m)
    {
        this->value[0] = detail::tvec2<valType>(m[0]);
        this->value[1] = detail::tvec2<valType>(m[1]);
        this->value[2] = detail::tvec2<valType>(m[2]);
        this->value[3] = detail::tvec2<valType>(m[3]);
	}

    template <typename valType> 
    inline tmat4x2<valType>::tmat4x2(tmat2x2<valType> const & m)
    {
        this->value[0] = detail::tvec2<valType>(m[0]);
        this->value[1] = detail::tvec2<valType>(m[1]);
        this->value[2] = detail::tvec2<valType>(valType(0));
        this->value[3] = detail::tvec2<valType>(valType(0));
    }

    template <typename valType> 
    inline tmat4x2<valType>::tmat4x2(const tmat3x3<valType>& m)
    {
        this->value[0] = detail::tvec2<valType>(m[0]);
        this->value[1] = detail::tvec2<valType>(m[1]);
        this->value[2] = detail::tvec2<valType>(m[2]);
        this->value[3] = detail::tvec2<valType>(valType(0));
    }

    template <typename valType> 
    inline tmat4x2<valType>::tmat4x2(const tmat4x4<valType>& m)
    {
        this->value[0] = detail::tvec2<valType>(m[0]);
        this->value[1] = detail::tvec2<valType>(m[1]);
        this->value[2] = detail::tvec2<valType>(m[2]);
        this->value[3] = detail::tvec2<valType>(m[3]);
    }

    template <typename valType> 
    inline tmat4x2<valType>::tmat4x2(const tmat2x3<valType>& m)
    {
        this->value[0] = detail::tvec2<valType>(m[0]);
        this->value[1] = detail::tvec2<valType>(m[1]);
        this->value[2] = detail::tvec2<valType>(valType(0));
        this->value[3] = detail::tvec2<valType>(valType(0));
    }

    template <typename valType> 
    inline tmat4x2<valType>::tmat4x2(const tmat3x2<valType>& m)
    {
        this->value[0] = detail::tvec2<valType>(m[0]);
        this->value[1] = detail::tvec2<valType>(m[1]);
        this->value[2] = detail::tvec2<valType>(m[2]);
        this->value[3] = detail::tvec2<valType>(valType(0));
    }

    template <typename valType> 
    inline tmat4x2<valType>::tmat4x2(const tmat2x4<valType>& m)
    {
        this->value[0] = detail::tvec2<valType>(m[0]);
        this->value[1] = detail::tvec2<valType>(m[1]);
        this->value[2] = detail::tvec2<valType>(valType(0));
        this->value[3] = detail::tvec2<valType>(valType(0));
    }

    template <typename valType> 
    inline tmat4x2<valType>::tmat4x2(const tmat4x3<valType>& m)
    {
        this->value[0] = detail::tvec2<valType>(m[0]);
        this->value[1] = detail::tvec2<valType>(m[1]);
        this->value[2] = detail::tvec2<valType>(m[2]);
        this->value[3] = detail::tvec2<valType>(m[3]);
    }

    template <typename valType> 
    inline tmat4x2<valType>::tmat4x2(const tmat3x4<valType>& m)
    {
        this->value[0] = detail::tvec2<valType>(m[0]);
        this->value[1] = detail::tvec2<valType>(m[1]);
        this->value[2] = detail::tvec2<valType>(m[2]);
        this->value[3] = detail::tvec2<valType>(valType(0));
    }

    //////////////////////////////////////////////////////////////
    // Unary updatable operators

    template <typename valType> 
    inline tmat4x2<valType>& tmat4x2<valType>::operator= (const tmat4x2<valType>& m)
    {
        this->value[0] = m[0];
        this->value[1] = m[1];
        this->value[2] = m[2];
        this->value[3] = m[3];
        return *this;
    }

    template <typename valType> 
    inline tmat4x2<valType>& tmat4x2<valType>::operator+= (const valType & s)
    {
        this->value[0] += s;
        this->value[1] += s;
        this->value[2] += s;
        this->value[3] += s;
        return *this;
    }

    template <typename valType> 
    inline tmat4x2<valType>& tmat4x2<valType>::operator+= (const tmat4x2<valType>& m)
    {
        this->value[0] += m[0];
        this->value[1] += m[1];
        this->value[2] += m[2];
        this->value[3] += m[3];
        return *this;
    }

    template <typename valType> 
    inline tmat4x2<valType>& tmat4x2<valType>::operator-= (const valType & s)
    {
        this->value[0] -= s;
        this->value[1] -= s;
        this->value[2] -= s;
        this->value[3] -= s;
        return *this;
    }

    template <typename valType> 
    inline tmat4x2<valType>& tmat4x2<valType>::operator-= (const tmat4x2<valType>& m)
    {
        this->value[0] -= m[0];
        this->value[1] -= m[1];
        this->value[2] -= m[2];
        this->value[3] -= m[3];
        return *this;
    }

    template <typename valType> 
    inline tmat4x2<valType>& tmat4x2<valType>::operator*= (const valType & s)
    {
        this->value[0] *= s;
        this->value[1] *= s;
        this->value[2] *= s;
        this->value[3] *= s;
        return *this;
    }

    template <typename valType> 
    inline tmat4x2<valType>& tmat4x2<valType>::operator*= (const tmat2x4<valType>& m)
    {
        return (*this = tmat4x2<valType>(*this * m));
    }

    template <typename valType> 
    inline tmat4x2<valType> & tmat4x2<valType>::operator/= (const valType & s)
    {
        this->value[0] /= s;
        this->value[1] /= s;
        this->value[2] /= s;
        this->value[3] /= s;
        return *this;
    }

    //template <typename valType> 
    //inline tmat2x2<valType>& tmat4x2<valType>::operator/= (const tmat2x4<valType>& m)
    //{
    //    return (*this = *this / m);
    //}

    template <typename valType> 
    inline tmat4x2<valType>& tmat4x2<valType>::operator++ ()
    {
        ++this->value[0];
        ++this->value[1];
        ++this->value[2];
        ++this->value[3];
        return *this;
    }

    template <typename valType> 
    inline tmat4x2<valType>& tmat4x2<valType>::operator-- ()
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
    inline tmat2x4<valType> tmat4x2<valType>::_inverse() const
    {
		assert(0); //g.truc.creation[at]gmail.com
    }

    //////////////////////////////////////////////////////////////
    // Binary operators

    template <typename valType> 
    inline tmat4x2<valType> operator+ (const tmat4x2<valType>& m, const valType & s)
    {
        return tmat4x2<valType>(
            m[0] + s,
            m[1] + s,
            m[2] + s,
            m[3] + s);
    }

    template <typename valType> 
    inline tmat4x2<valType> operator+ (const tmat4x2<valType>& m1, const tmat4x2<valType>& m2)
    {
        return tmat4x2<valType>(
            m1[0] + m2[0],
            m1[1] + m2[1],
            m1[2] + m2[2],
            m1[3] + m2[3]);
    }

    template <typename valType> 
    inline tmat4x2<valType> operator- (const tmat4x2<valType>& m, const valType & s)
    {
        return tmat4x2<valType>(
            m[0] - s,
            m[1] - s,
            m[2] - s,
            m[3] - s);
    }

    template <typename valType> 
    inline tmat4x2<valType> operator- (const tmat4x2<valType>& m1, const tmat4x2<valType>& m2)
    {
        return tmat4x2<valType>(
            m1[0] - m2[0],
            m1[1] - m2[1],
            m1[2] - m2[2],
            m1[3] - m2[3]);
    }

    template <typename valType> 
    inline tmat4x2<valType> operator* (const tmat4x2<valType>& m, const valType & s)
    {
        return tmat4x2<valType>(
            m[0] * s,
            m[1] * s,
            m[2] * s,
            m[3] * s);
    }

    template <typename valType> 
    inline tmat4x2<valType> operator* (const valType & s, const tmat4x2<valType> & m)
    {
        return tmat4x2<valType>(
            m[0] * s,
            m[1] * s,
            m[2] * s,
            m[3] * s);
    }
   
    template <typename valType>
    inline detail::tvec2<valType> operator* (const tmat4x2<valType>& m, const tvec4<valType>& v)
    {
        return detail::tvec2<valType>(
            m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w,
            m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w);
    }

    template <typename valType> 
    inline tvec4<valType> operator* (const detail::tvec2<valType>& v, const tmat4x2<valType>& m) 
    {
        return tvec4<valType>(
            v.x * m[0][0] + v.y * m[0][1],
            v.x * m[1][0] + v.y * m[1][1],
            v.x * m[2][0] + v.y * m[2][1],
            v.x * m[3][0] + v.y * m[3][1]);
    }

    template <typename valType> 
    inline tmat2x2<valType> operator* (const tmat4x2<valType>& m1, const tmat2x4<valType>& m2)
    {
        const valType SrcA00 = m1[0][0];
        const valType SrcA01 = m1[0][1];
        const valType SrcA10 = m1[1][0];
        const valType SrcA11 = m1[1][1];
        const valType SrcA20 = m1[2][0];
        const valType SrcA21 = m1[2][1];
        const valType SrcA30 = m1[3][0];
        const valType SrcA31 = m1[3][1];

        const valType SrcB00 = m2[0][0];
        const valType SrcB01 = m2[0][1];
        const valType SrcB02 = m2[0][2];
        const valType SrcB03 = m2[0][3];
        const valType SrcB10 = m2[1][0];
        const valType SrcB11 = m2[1][1];
        const valType SrcB12 = m2[1][2];
        const valType SrcB13 = m2[1][3];

        tmat2x2<valType> Result;
        Result[0][0] = SrcA00 * SrcB00 + SrcA01 * SrcB01 + SrcA20 * SrcB02 + SrcA30 * SrcB03;
        Result[0][1] = SrcA01 * SrcB00 + SrcA11 * SrcB01 + SrcA21 * SrcB02 + SrcA31 * SrcB03;
        Result[1][0] = SrcA00 * SrcB10 + SrcA10 * SrcB11 + SrcA20 * SrcB12 + SrcA30 * SrcB13;
        Result[1][1] = SrcA01 * SrcB10 + SrcA11 * SrcB11 + SrcA21 * SrcB12 + SrcA31 * SrcB13;
        return Result;
    }

    template <typename valType> 
    inline tmat4x2<valType> operator/ (const tmat4x2<valType>& m, const valType & s)
    {
        return tmat4x2<valType>(
            m[0] / s,
            m[1] / s,
            m[2] / s,
            m[3] / s);        
    }

    template <typename valType> 
    inline tmat4x2<valType> operator/ (const valType & s, const tmat4x2<valType>& m)
    {
        return tmat4x2<valType>(
            s / m[0],
            s / m[1],
            s / m[2],
            s / m[3]);        
    }

	//template <typename valType> 
	//tvec2<valType> operator/ 
	//(
	//	tmat4x2<valType> const & m, 
	//	tvec4<valType> const & v
	//)
	//{
	//	return m._inverse() * v;
	//}

	//template <typename valType> 
	//tvec4<valType> operator/ 
	//(
	//	tvec2<valType> const & v, 
	//	tmat4x2<valType> const & m
	//)
	//{
	//	return v * m._inverse();
	//}

	//template <typename valType> 
	//inline tmat2x2<valType> operator/ 
	//(
	//	tmat4x2<valType> const & m1, 
	//	tmat2x4<valType> const & m2
	//)
	//{
	//	return m1 * m2._inverse();
	//}

	// Unary constant operators
    template <typename valType> 
    inline tmat4x2<valType> const operator- 
	(
		tmat4x2<valType> const & m
	)
    {
        return tmat4x2<valType>(
            -m[0], 
            -m[1], 
            -m[2], 
            -m[3]);
    }

    template <typename valType> 
    inline tmat4x2<valType> const operator++ 
	(
		tmat4x2<valType> const & m, 
		int
	) 
    {
        return tmat4x2<valType>(
            m[0] + valType(1),
            m[1] + valType(1),
            m[2] + valType(1),
            m[3] + valType(1));
    }

    template <typename valType> 
    inline tmat4x2<valType> const operator-- 
	(
		tmat4x2<valType> const & m, 
		int
	) 
    {
        return tmat4x2<valType>(
            m[0] - valType(1),
            m[1] - valType(1),
            m[2] - valType(1),
            m[3] - valType(1));
    }

} //namespace detail
} //namespace glm
