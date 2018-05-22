///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2005-01-27
// Updated : 2008-08-25
// Licence : This source is under MIT License
// File    : glm/core/type_mat3x3.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail{

	template <typename valType> 
	typename tmat3x3<valType>::size_type tmat3x3<valType>::value_size()
	{
		return typename tmat3x3<valType>::size_type(3);
	}

	template <typename valType> 
	typename tmat3x3<valType>::size_type tmat3x3<valType>::col_size()
	{
		return typename tmat3x3<valType>::size_type(3);
	}

	template <typename valType> 
	typename tmat3x3<valType>::size_type tmat3x3<valType>::row_size()
	{
		return typename tmat3x3<valType>::size_type(3);
	}

	template <typename valType> 
	bool tmat3x3<valType>::is_matrix()
	{
		return true;
	}

	//////////////////////////////////////
	// Accesses

	template <typename valType>
	detail::tvec3<valType>& tmat3x3<valType>::operator[](typename tmat3x3<valType>::size_type i)
	{
		assert(
			i >= typename tmat3x3<valType>::size_type(0) && 
			i < tmat3x3<valType>::col_size());

		return value[i];
	}

	template <typename valType>
	const detail::tvec3<valType>& tmat3x3<valType>::operator[](typename tmat3x3<valType>::size_type i) const
	{
		assert(
			i >= typename tmat3x3<valType>::size_type(0) &&
			i < tmat3x3<valType>::col_size());

		return value[i];
	}

    //////////////////////////////////////////////////////////////
    // mat3 constructors

    template <typename T> 
    inline tmat3x3<T>::tmat3x3()
    {
        this->value[0] = detail::tvec3<T>(1, 0, 0);
        this->value[1] = detail::tvec3<T>(0, 1, 0);
        this->value[2] = detail::tvec3<T>(0, 0, 1);
    }

    template <typename T> 
    inline tmat3x3<T>::tmat3x3(const T f)
    {
        this->value[0] = detail::tvec3<T>(f, 0, 0);
        this->value[1] = detail::tvec3<T>(0, f, 0);
        this->value[2] = detail::tvec3<T>(0, 0, f);
    }

    template <typename T> 
    inline tmat3x3<T>::tmat3x3
    (
        const T x0, const T y0, const T z0, 
        const T x1, const T y1, const T z1,
        const T x2, const T y2, const T z2
    )
    {
        this->value[0] = detail::tvec3<T>(x0, y0, z0);
        this->value[1] = detail::tvec3<T>(x1, y1, z1);
        this->value[2] = detail::tvec3<T>(x2, y2, z2);
    }

    template <typename T> 
    inline tmat3x3<T>::tmat3x3
    (
        const detail::tvec3<T>& v0, 
        const detail::tvec3<T>& v1, 
        const detail::tvec3<T>& v2
    )
    {
        this->value[0] = v0;
        this->value[1] = v1;
        this->value[2] = v2;
    }

    //////////////////////////////////////////////////////////////
    // mat3 conversions

    template <typename T> 
    template <typename U> 
    inline tmat3x3<T>::tmat3x3(const tmat3x3<U>& m)
    {
        this->value[0] = detail::tvec3<T>(m[0]);
        this->value[1] = detail::tvec3<T>(m[1]);
        this->value[2] = detail::tvec3<T>(m[2]);
	}

    template <typename T> 
    inline tmat3x3<T>::tmat3x3(tmat2x2<T> const & m)
    {
        this->value[0] = detail::tvec3<T>(m[0], T(0));
        this->value[1] = detail::tvec3<T>(m[1], T(0));
        this->value[2] = detail::tvec3<T>(detail::tvec2<T>(0), T(1));
    }

    template <typename T> 
    inline tmat3x3<T>::tmat3x3(const tmat4x4<T>& m)
    {
        this->value[0] = detail::tvec3<T>(m[0]);
        this->value[1] = detail::tvec3<T>(m[1]);
        this->value[2] = detail::tvec3<T>(m[2]);
    }

	template <typename T> 
    inline tmat3x3<T>::tmat3x3(const tmat2x3<T>& m)
    {
        this->value[0] = m[0];
        this->value[1] = m[1];
        this->value[2] = detail::tvec3<T>(detail::tvec2<T>(0), T(1));
    }

    template <typename T> 
    inline tmat3x3<T>::tmat3x3(const tmat3x2<T>& m)
    {
        this->value[0] = detail::tvec3<T>(m[0], T(0));
        this->value[1] = detail::tvec3<T>(m[1], T(0));
        this->value[2] = detail::tvec3<T>(m[2], T(1));
    }

    template <typename T> 
    inline tmat3x3<T>::tmat3x3(const tmat2x4<T>& m)
    {
        this->value[0] = detail::tvec3<T>(m[0]);
        this->value[1] = detail::tvec3<T>(m[1]);
        this->value[2] = detail::tvec3<T>(detail::tvec2<T>(0), T(1));
    }

    template <typename T> 
    inline tmat3x3<T>::tmat3x3(const tmat4x2<T>& m)
    {
        this->value[0] = detail::tvec3<T>(m[0], T(0));
        this->value[1] = detail::tvec3<T>(m[1], T(0));
        this->value[2] = detail::tvec3<T>(m[2], T(1));
    }

    template <typename T> 
    inline tmat3x3<T>::tmat3x3(const tmat3x4<T>& m)
    {
        this->value[0] = detail::tvec3<T>(m[0]);
        this->value[1] = detail::tvec3<T>(m[1]);
        this->value[2] = detail::tvec3<T>(m[2]);
    }

    template <typename T> 
    inline tmat3x3<T>::tmat3x3(const tmat4x3<T>& m)
    {
        this->value[0] = m[0];
        this->value[1] = m[1];
        this->value[2] = m[2];
    }

/*
    template <typename T> 
    inline tmat3x3<T>::tmat3x3(const T* a)
    {
        this->value[0] = detail::tvec3<T>(a[0], a[1], a[2]);
        this->value[1] = detail::tvec3<T>(a[3], a[4], a[5]);
        this->value[2] = detail::tvec3<T>(a[6], a[7], a[8]);
    }
*/
    /*
    // GL_GTX_euler_angles
    template <typename T> 
    inline tmat3x3<T>::tmat3x3(const detail::tvec3<T> & angles)
    {
        T ch = cos(angles.x);
        T sh = sin(angles.x);
        T cp = cos(angles.y);
        T sp = sin(angles.y);
        T cb = cos(angles.z);
        T sb = sin(angles.z);

        value[0][0] = ch * cb + sh * sp * sb;
        value[0][1] = sb * cp;
        value[0][2] = -sh * cb + ch * sp * sb;
        value[1][0] = -ch * sb + sh * sp * cb;
        value[1][1] = cb * cp;
        value[1][2] = sb * sh + ch * sp * cb;
        value[2][0] = sh * cp;
        value[2][1] = -sp;
        value[2][2] = ch * cp;
    }
    */
    //////////////////////////////////////////////////////////////
    // mat3 conversions
    /*
    template <typename T> 
    inline tmat3x3<T>::tmat3x3(const tquat<T> & q)
    {
        this->value[0][0] = 1 - 2 * q.y * q.y - 2 * q.z * q.z;
        this->value[0][1] = 2 * q.x * q.y + 2 * q.w * q.z;
        this->value[0][2] = 2 * q.x * q.z - 2 * q.w * q.y;

        this->value[1][0] = 2 * q.x * q.y - 2 * q.w * q.z;
        this->value[1][1] = 1 - 2 * q.x * q.x - 2 * q.z * q.z;
        this->value[1][2] = 2 * q.y * q.z + 2 * q.w * q.x;

        this->value[2][0] = 2 * q.x * q.z + 2 * q.w * q.y;
        this->value[2][1] = 2 * q.y * q.z - 2 * q.w * q.x;
        this->value[2][2] = 1 - 2 * q.x * q.x - 2 * q.y * q.y;
    }
    */
    //////////////////////////////////////////////////////////////
    // mat3 operators

    template <typename T> 
    inline tmat3x3<T>& tmat3x3<T>::operator=(const tmat3x3<T>& m)
    {
	    this->value[0] = m[0];
	    this->value[1] = m[1];
        this->value[2] = m[2];
	    return *this;
    }

    template <typename T> 
    inline tmat3x3<T>& tmat3x3<T>::operator+= (const T & s)
    {
	    this->value[0] += s;
	    this->value[1] += s;
        this->value[2] += s;
	    return *this;
    }

    template <typename T> 
    inline tmat3x3<T>& tmat3x3<T>::operator+= (const tmat3x3<T>& m)
    {
	    this->value[0] += m[0];
	    this->value[1] += m[1];
        this->value[2] += m[2];
	    return *this;
    }

    template <typename T> 
    inline tmat3x3<T>& tmat3x3<T>::operator-= (const T & s)
    {
	    this->value[0] -= s;
	    this->value[1] -= s;
        this->value[2] -= s;
	    return *this;
    }

    template <typename T> 
    inline tmat3x3<T>& tmat3x3<T>::operator-= (const tmat3x3<T>& m)
    {
	    this->value[0] -= m[0];
	    this->value[1] -= m[1];
        this->value[2] -= m[2];
	    return *this;
    }

    template <typename T> 
    inline tmat3x3<T>& tmat3x3<T>::operator*= (const T & s)
    {
	    this->value[0] *= s;
	    this->value[1] *= s;
        this->value[2] *= s;
	    return *this;
    }

    template <typename T> 
    inline tmat3x3<T>& tmat3x3<T>::operator*= (const tmat3x3<T>& m)
    {
        return (*this = *this * m);
    }

    template <typename T> 
    inline tmat3x3<T>& tmat3x3<T>::operator/= (const T & s)
    {
	    this->value[0] /= s;
	    this->value[1] /= s;
        this->value[2] /= s;
	    return *this;
    }

    template <typename T> 
    inline tmat3x3<T>& tmat3x3<T>::operator/= (const tmat3x3<T>& m)
    {
        return (*this = *this / m);
    }

    template <typename T> 
    inline tmat3x3<T>& tmat3x3<T>::operator++ ()
    {
	    this->value[0]++;
	    this->value[1]++;
        this->value[2]++;
	    return *this;
    }

    template <typename T> 
    inline tmat3x3<T>& tmat3x3<T>::operator-- ()
    {
	    this->value[0]--;
	    this->value[1]--;
        this->value[2]--;
	    return *this;
    }

    template <typename T> 
    inline tmat3x3<T> tmat3x3<T>::_inverse() const
    {
        T S00 = value[0][0];
        T S01 = value[0][1];
        T S02 = value[0][2];

        T S10 = value[1][0];
        T S11 = value[1][1];
        T S12 = value[1][2];

        T S20 = value[2][0];
        T S21 = value[2][1];
        T S22 = value[2][2];

        tmat3x3<T> Inverse(
            + (S11 * S22 - S21 * S12),
            - (S10 * S22 - S20 * S12),
            + (S10 * S21 - S20 * S11),
            - (S01 * S22 - S21 * S02),
            + (S00 * S22 - S20 * S02),
            - (S00 * S21 - S20 * S01),
            + (S01 * S12 - S11 * S02),
            - (S00 * S12 - S10 * S02),
            + (S00 * S11 - S10 * S01));

        T Determinant = S00 * (S11 * S22 - S21 * S12)
                      - S10 * (S01 * S22 - S21 * S02)
                      + S20 * (S01 * S12 - S11 * S02);

        Inverse /= Determinant;
        return Inverse;
    }

    //////////////////////////////////////////////////////////////
	// Binary operators

    template <typename T> 
    inline tmat3x3<T> operator+ (const tmat3x3<T>& m, const T & s)
    {
        return tmat3x3<T>(
            m[0] + s,
            m[1] + s,
            m[2] + s);
    }

    template <typename T> 
    inline tmat3x3<T> operator+ (const T & s, const tmat3x3<T>& m)
    {
        return tmat3x3<T>(
            m[0] + s,
            m[1] + s,
            m[2] + s);
    }

    //template <typename T> 
    //inline tvec3<T> operator+ (const tmat3x3<T>& m, const tvec3<T>& v)
    //{

    //}

    //template <typename T> 
    //inline tvec3<T> operator+ (const tvec3<T>& v, const tmat3x3<T>& m)
    //{

    //}

    template <typename T> 
    inline tmat3x3<T> operator+ (const tmat3x3<T>& m1, const tmat3x3<T>& m2)
    {
        return tmat3x3<T>(
            m1[0] + m2[0],
            m1[1] + m2[1],
            m1[2] + m2[2]);
    }

    template <typename T> 
    inline tmat3x3<T> operator- (const tmat3x3<T>& m, const T & s)
    {
        return tmat3x3<T>(
            m[0] - s,
            m[1] - s,
            m[2] - s);
    }

    template <typename T> 
    inline tmat3x3<T> operator- (const T & s, const tmat3x3<T>& m)
    {
        return tmat3x3<T>(
            s - m[0],
            s - m[1],
            s - m[2]);
    }

    //template <typename T> 
    //inline detail::tvec3<T> operator- (const tmat3x3<T>& m, const detail::tvec3<T>& v)
    //{

    //}

    //template <typename T> 
    //inline detail::tvec3<T> operator- (const detail::tvec3<T>& v, const tmat3x3<T>& m)
    //{

    //}

    template <typename T> 
    inline tmat3x3<T> operator- (const tmat3x3<T>& m1, const tmat3x3<T>& m2)
    {
        return tmat3x3<T>(
            m1[0] - m2[0],
            m1[1] - m2[1],
            m1[2] - m2[2]);
    }

    template <typename T> 
    inline tmat3x3<T> operator* (const tmat3x3<T>& m, const T & s)
    {
        return tmat3x3<T>(
            m[0] * s,
            m[1] * s,
            m[2] * s);
    }

    template <typename T> 
    inline tmat3x3<T> operator* (const T & s, const tmat3x3<T>& m)
    {
        return tmat3x3<T>(
            m[0] * s,
            m[1] * s,
            m[2] * s);
    }

    template <typename T> 
    inline detail::tvec3<T> operator* (const tmat3x3<T>& m, const detail::tvec3<T>& v)
    {
        return detail::tvec3<T>(
            m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z,
            m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z,
            m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z);
    }

    template <typename T> 
    inline detail::tvec3<T> operator* (const detail::tvec3<T>& v, const tmat3x3<T>& m)
    {
        return detail::tvec3<T>(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z);
    }

    template <typename T> 
    inline tmat3x3<T> operator* (const tmat3x3<T>& m1, const tmat3x3<T>& m2)
    {
        const T SrcA00 = m1[0][0];
        const T SrcA01 = m1[0][1];
        const T SrcA02 = m1[0][2];
        const T SrcA10 = m1[1][0];
        const T SrcA11 = m1[1][1];
        const T SrcA12 = m1[1][2];
        const T SrcA20 = m1[2][0];
        const T SrcA21 = m1[2][1];
        const T SrcA22 = m1[2][2];

        const T SrcB00 = m2[0][0];
        const T SrcB01 = m2[0][1];
        const T SrcB02 = m2[0][2];
        const T SrcB10 = m2[1][0];
        const T SrcB11 = m2[1][1];
        const T SrcB12 = m2[1][2];
        const T SrcB20 = m2[2][0];
        const T SrcB21 = m2[2][1];
        const T SrcB22 = m2[2][2];

        tmat3x3<T> Result;
        Result[0][0] = SrcA00 * SrcB00 + SrcA10 * SrcB01 + SrcA20 * SrcB02;
        Result[0][1] = SrcA01 * SrcB00 + SrcA11 * SrcB01 + SrcA21 * SrcB02;
        Result[0][2] = SrcA02 * SrcB00 + SrcA12 * SrcB01 + SrcA22 * SrcB02;
        Result[1][0] = SrcA00 * SrcB10 + SrcA10 * SrcB11 + SrcA20 * SrcB12;
        Result[1][1] = SrcA01 * SrcB10 + SrcA11 * SrcB11 + SrcA21 * SrcB12;
        Result[1][2] = SrcA02 * SrcB10 + SrcA12 * SrcB11 + SrcA22 * SrcB12;
        Result[2][0] = SrcA00 * SrcB20 + SrcA10 * SrcB21 + SrcA20 * SrcB22;
        Result[2][1] = SrcA01 * SrcB20 + SrcA11 * SrcB21 + SrcA21 * SrcB22;
        Result[2][2] = SrcA02 * SrcB20 + SrcA12 * SrcB21 + SrcA22 * SrcB22;
        return Result;
    }


    template <typename T> 
    inline tmat3x3<T> operator/ (const tmat3x3<T>& m, const T & s)
    {
        return tmat3x3<T>(
            m[0] / s,
            m[1] / s,
            m[2] / s);
    }

    template <typename T> 
    inline tmat3x3<T> operator/ (const T & s, const tmat3x3<T>& m)
    {
        return tmat3x3<T>(
            s / m[0],
            s / m[1],
            s / m[2]);
    }

	template <typename T> 
	inline detail::tvec3<T> operator/ 
	(
		tmat3x3<T> const & m, 
		tvec3<T> const & v
	)
	{
		return m._inverse() * v;
	}

	template <typename T> 
	inline detail::tvec3<T> operator/ 
	(
		tvec3<T> const & v, 
		tmat3x3<T> const & m
	)
	{
		return v * m._inverse();
	}

	template <typename T> 
	inline tmat3x3<T> operator/ 
	(
		tmat3x3<T> const & m1, 
		tmat3x3<T> const & m2
	)
	{
		return m1 * m2._inverse();
	}

	// Unary constant operators
    template <typename valType> 
    inline tmat3x3<valType> const operator- 
	(
		tmat3x3<valType> const & m
	)
    {
        return tmat3x3<valType>(
            -m[0], 
            -m[1],
            -m[2]);
    }

    template <typename valType> 
    inline tmat3x3<valType> const operator++ 
	(
		tmat3x3<valType> const & m, 
		int
	) 
    {
        return tmat3x3<valType>(
            m[0] + valType(1),
            m[1] + valType(1),
            m[2] + valType(1));
    }

    template <typename valType> 
    inline tmat3x3<valType> const operator-- 
	(
		tmat3x3<valType> const & m, 
		int
	) 
    {
        return tmat3x3<valType>(
            m[0] - valType(1),
            m[1] - valType(1),
            m[2] - valType(1));
    }

} //namespace detail
} //namespace glm
