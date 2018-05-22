///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2005-01-27
// Updated : 2008-09-09
// Licence : This source is under MIT License
// File    : glm/core/type_mat4x4.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail{

	template <typename valType> 
	typename tmat4x4<valType>::size_type tmat4x4<valType>::value_size()
	{
		return typename tmat4x4<valType>::size_type(4);
	}

	template <typename valType> 
	typename tmat4x4<valType>::size_type tmat4x4<valType>::col_size()
	{
		return typename tmat4x4<valType>::size_type(4);
	}

	template <typename valType> 
	typename tmat4x4<valType>::size_type tmat4x4<valType>::row_size()
	{
		return typename tmat4x4<valType>::size_type(4);
	}

	template <typename valType> 
	bool tmat4x4<valType>::is_matrix()
	{
		return true;
	}

	//////////////////////////////////////
	// Accesses

	template <typename valType>
	detail::tvec4<valType>& tmat4x4<valType>::operator[]
	(
		typename tmat4x4<valType>::size_type i
	)
	{
		assert(
			i >= typename tmat4x4<valType>::size_type(0) && 
			i < tmat4x4<valType>::col_size());

		return value[i];
	}

	template <typename valType>
	const detail::tvec4<valType>& tmat4x4<valType>::operator[]
	(
		typename tmat4x4<valType>::size_type i
	) const
	{
		assert(
			i >= typename tmat4x4<valType>::size_type(0) && 
			i < tmat4x4<valType>::col_size());

		return value[i];
	}

    //////////////////////////////////////////////////////////////
    // mat4 constructors

    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4()
    {
        this->value[0] = detail::tvec4<valType>(1, 0, 0, 0);
        this->value[1] = detail::tvec4<valType>(0, 1, 0, 0);
        this->value[2] = detail::tvec4<valType>(0, 0, 1, 0);
        this->value[3] = detail::tvec4<valType>(0, 0, 0, 1);
    }

    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4(typename tmat4x4<valType>::ctor)
    {}

    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4(valType const & f)
    {
        this->value[0] = detail::tvec4<valType>(f, 0, 0, 0);
        this->value[1] = detail::tvec4<valType>(0, f, 0, 0);
        this->value[2] = detail::tvec4<valType>(0, 0, f, 0);
        this->value[3] = detail::tvec4<valType>(0, 0, 0, f);
    }

    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4
    (
        const valType x0, const valType y0, const valType z0, const valType w0,
        const valType x1, const valType y1, const valType z1, const valType w1,
        const valType x2, const valType y2, const valType z2, const valType w2,
        const valType x3, const valType y3, const valType z3, const valType w3
    )
    {
        this->value[0] = detail::tvec4<valType>(x0, y0, z0, w0);
        this->value[1] = detail::tvec4<valType>(x1, y1, z1, w1);
        this->value[2] = detail::tvec4<valType>(x2, y2, z2, w2);
        this->value[3] = detail::tvec4<valType>(x3, y3, z3, w3);
    }

    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4
    (
        detail::tvec4<valType> const & v0, 
        detail::tvec4<valType> const & v1, 
        detail::tvec4<valType> const & v2,
        detail::tvec4<valType> const & v3
    )
    {
        this->value[0] = v0;
        this->value[1] = v1;
        this->value[2] = v2;
        this->value[3] = v3;
    }

    template <typename valType> 
    template <typename U> 
    inline tmat4x4<valType>::tmat4x4
	(
		tmat4x4<U> const & m
	)
    {
        this->value[0] = detail::tvec4<valType>(m[0]);
        this->value[1] = detail::tvec4<valType>(m[1]);
        this->value[2] = detail::tvec4<valType>(m[2]);
        this->value[3] = detail::tvec4<valType>(m[3]);
	}

    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4
	(
		tmat2x2<valType> const & m
	)
    {
        this->value[0] = detail::tvec4<valType>(m[0], detail::tvec2<valType>(0));
        this->value[1] = detail::tvec4<valType>(m[1], detail::tvec2<valType>(0));
        this->value[2] = detail::tvec4<valType>(valType(0));
        this->value[3] = detail::tvec4<valType>(valType(0), valType(0), valType(0), valType(1));
    }

    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4
	(
		tmat3x3<valType> const & m
	)
    {
        this->value[0] = detail::tvec4<valType>(m[0], valType(0));
        this->value[1] = detail::tvec4<valType>(m[1], valType(0));
        this->value[2] = detail::tvec4<valType>(m[2], valType(0));
        this->value[3] = detail::tvec4<valType>(valType(0), valType(0), valType(0), valType(1));
    }

	template <typename valType> 
    inline tmat4x4<valType>::tmat4x4
	(
		tmat2x3<valType> const & m
	)
    {
        this->value[0] = detail::tvec4<valType>(m[0], valType(0));
        this->value[1] = detail::tvec4<valType>(m[1], valType(0));
        this->value[2] = detail::tvec4<valType>(valType(0));
        this->value[3] = detail::tvec4<valType>(valType(0), valType(0), valType(0), valType(1));
    }

    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4
	(
		tmat3x2<valType> const & m
	)
    {
        this->value[0] = detail::tvec4<valType>(m[0], detail::tvec2<valType>(0));
        this->value[1] = detail::tvec4<valType>(m[1], detail::tvec2<valType>(0));
        this->value[2] = detail::tvec4<valType>(m[2], detail::tvec2<valType>(0));
        this->value[3] = detail::tvec4<valType>(valType(0), valType(0), valType(0), valType(1));
    }

    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4
	(
		tmat2x4<valType> const & m
	)
    {
        this->value[0] = m[0];
        this->value[1] = m[1];
        this->value[2] = detail::tvec4<valType>(valType(0));
        this->value[3] = detail::tvec4<valType>(valType(0), valType(0), valType(0), valType(1));
    }

    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4
	(
		tmat4x2<valType> const & m
	)
    {
        this->value[0] = detail::tvec4<valType>(m[0], detail::tvec2<valType>(0));
        this->value[1] = detail::tvec4<valType>(m[1], detail::tvec2<valType>(0));
        this->value[2] = detail::tvec4<valType>(valType(0));
        this->value[3] = detail::tvec4<valType>(valType(0), valType(0), valType(0), valType(1));
    }

    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4
	(
		tmat3x4<valType> const & m
	)
    {
        this->value[0] = m[0];
        this->value[1] = m[1];
        this->value[2] = m[2];
        this->value[3] = detail::tvec4<valType>(valType(0), valType(0), valType(0), valType(1));
    }

    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4
	(
		tmat4x3<valType> const & m
	)
    {
        this->value[0] = detail::tvec4<valType>(m[0], valType(0));
        this->value[1] = detail::tvec4<valType>(m[1], valType(0));
        this->value[2] = detail::tvec4<valType>(m[2], valType(0));
        this->value[3] = detail::tvec4<valType>(m[3], valType(1));
    }

/*
    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4(const valType* a)
    {
        this->value[0] = detail::tvec4<valType>(a[0], a[1], a[2], a[3]);
        this->value[1] = detail::tvec4<valType>(a[4], a[5], a[6], a[7]);
        this->value[2] = detail::tvec4<valType>(a[8], a[9], a[10], a[11]);
        this->value[4] = detail::tvec4<valType>(a[12], a[13], a[14], a[15]);
    }
*/
    /*
    // GL_GTX_euler_angles
    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4(const tvec3<valType> & angles)
    {
        valType ch = cos(angles.x);
        valType sh = sin(angles.x);
        valType cp = cos(angles.y);
        valType sp = sin(angles.y);
        valType cb = cos(angles.z);
        valType sb = sin(angles.z);

        value[0][0] = ch * cb + sh * sp * sb;
        value[0][1] = sb * cp;
        value[0][2] = -sh * cb + ch * sp * sb;
        value[0][3] = 0.0f;
        value[1][0] = -ch * sb + sh * sp * cb;
        value[1][1] = cb * cp;
        value[1][2] = sb * sh + ch * sp * cb;
        value[1][3] = 0.0f;
        value[2][0] = sh * cp;
        value[2][1] = -sp;
        value[2][2] = ch * cp;
        value[2][3] = 0.0f;
        value[3][0] = 0.0f;
        value[3][1] = 0.0f;
        value[3][2] = 0.0f;
        value[3][3] = 1.0f;
    }
    */
    //////////////////////////////////////////////////////////////
    // mat4 conversion
    /*
    template <typename valType> 
    inline tmat4x4<valType>::tmat4x4(const tquat<valType> & q)
    {
        *this = tmat4x4<valType>(1);
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
    // mat4 operators

    template <typename valType> 
    inline tmat4x4<valType>& tmat4x4<valType>::operator= 
	(
		tmat4x4<valType> const & m
	)
    {
        //memcpy could be faster
        //memcpy(&this->value, &m.value, 16 * sizeof(valType));
        this->value[0] = m[0];
        this->value[1] = m[1];
        this->value[2] = m[2];
        this->value[3] = m[3];
        return *this;
    }

    template <typename valType> 
    inline tmat4x4<valType>& tmat4x4<valType>::operator+= 
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
    inline tmat4x4<valType>& tmat4x4<valType>::operator+= 
	(
		tmat4x4<valType> const & m
	)
    {
        this->value[0] += m[0];
        this->value[1] += m[1];
        this->value[2] += m[2];
        this->value[3] += m[3];
        return *this;
    }

    template <typename valType> 
    inline tmat4x4<valType>& tmat4x4<valType>::operator-= 
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
    inline tmat4x4<valType>& tmat4x4<valType>::operator-= 
	(
		tmat4x4<valType> const & m
	)
    {
        this->value[0] -= m[0];
        this->value[1] -= m[1];
        this->value[2] -= m[2];
        this->value[3] -= m[3];
        return *this;
    }

    template <typename valType> 
    inline tmat4x4<valType>& tmat4x4<valType>::operator*= 
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
    inline tmat4x4<valType>& tmat4x4<valType>::operator*= 
	(
		tmat4x4<valType> const & m
	)
    {
        return (*this = *this * m);
    }

    template <typename valType> 
    inline tmat4x4<valType> & tmat4x4<valType>::operator/= 
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

    template <typename valType> 
    inline tmat4x4<valType>& tmat4x4<valType>::operator/= 
	(
		tmat4x4<valType> const & m
	)
    {
        return (*this = *this / m);
    }

    template <typename valType> 
    inline tmat4x4<valType>& tmat4x4<valType>::operator++ ()
    {
        ++this->value[0];
        ++this->value[1];
        ++this->value[2];
        ++this->value[3];
        return *this;
    }

    template <typename valType> 
    inline tmat4x4<valType>& tmat4x4<valType>::operator-- ()
    {
        --this->value[0];
        --this->value[1];
        --this->value[2];
        --this->value[3];
        return *this;
    }

    template <typename valType> 
    inline tmat4x4<valType> tmat4x4<valType>::_inverse() const
    {
        // Calculate all mat2 determinants
        valType SubFactor00 = this->value[2][2] * this->value[3][3] - this->value[3][2] * this->value[2][3];
        valType SubFactor01 = this->value[2][1] * this->value[3][3] - this->value[3][1] * this->value[2][3];
        valType SubFactor02 = this->value[2][1] * this->value[3][2] - this->value[3][1] * this->value[2][2];
        valType SubFactor03 = this->value[2][0] * this->value[3][3] - this->value[3][0] * this->value[2][3];
        valType SubFactor04 = this->value[2][0] * this->value[3][2] - this->value[3][0] * this->value[2][2];
        valType SubFactor05 = this->value[2][0] * this->value[3][1] - this->value[3][0] * this->value[2][1];
        valType SubFactor06 = this->value[1][2] * this->value[3][3] - this->value[3][2] * this->value[1][3];
        valType SubFactor07 = this->value[1][1] * this->value[3][3] - this->value[3][1] * this->value[1][3];
        valType SubFactor08 = this->value[1][1] * this->value[3][2] - this->value[3][1] * this->value[1][2];
        valType SubFactor09 = this->value[1][0] * this->value[3][3] - this->value[3][0] * this->value[1][3];
        valType SubFactor10 = this->value[1][0] * this->value[3][2] - this->value[3][0] * this->value[1][2];
        valType SubFactor11 = this->value[1][1] * this->value[3][3] - this->value[3][1] * this->value[1][3];
        valType SubFactor12 = this->value[1][0] * this->value[3][1] - this->value[3][0] * this->value[1][1];
        valType SubFactor13 = this->value[1][2] * this->value[2][3] - this->value[2][2] * this->value[1][3];
        valType SubFactor14 = this->value[1][1] * this->value[2][3] - this->value[2][1] * this->value[1][3];
        valType SubFactor15 = this->value[1][1] * this->value[2][2] - this->value[2][1] * this->value[1][2];
        valType SubFactor16 = this->value[1][0] * this->value[2][3] - this->value[2][0] * this->value[1][3];
        valType SubFactor17 = this->value[1][0] * this->value[2][2] - this->value[2][0] * this->value[1][2];
        valType SubFactor18 = this->value[1][0] * this->value[2][1] - this->value[2][0] * this->value[1][1];

        tmat4x4<valType> Inverse(
            + (this->value[1][1] * SubFactor00 - this->value[1][2] * SubFactor01 + this->value[1][3] * SubFactor02),
            - (this->value[1][0] * SubFactor00 - this->value[1][2] * SubFactor03 + this->value[1][3] * SubFactor04),
            + (this->value[1][0] * SubFactor01 - this->value[1][1] * SubFactor03 + this->value[1][3] * SubFactor05),
            - (this->value[1][0] * SubFactor02 - this->value[1][1] * SubFactor04 + this->value[1][2] * SubFactor05),

            - (this->value[0][1] * SubFactor00 - this->value[0][2] * SubFactor01 + this->value[0][3] * SubFactor02),
            + (this->value[0][0] * SubFactor00 - this->value[0][2] * SubFactor03 + this->value[0][3] * SubFactor04),
            - (this->value[0][0] * SubFactor01 - this->value[0][1] * SubFactor03 + this->value[0][3] * SubFactor05),
            + (this->value[0][0] * SubFactor02 - this->value[0][1] * SubFactor04 + this->value[0][2] * SubFactor05),

            + (this->value[0][1] * SubFactor06 - this->value[0][2] * SubFactor07 + this->value[0][3] * SubFactor08),
            - (this->value[0][0] * SubFactor06 - this->value[0][2] * SubFactor09 + this->value[0][3] * SubFactor10),
            + (this->value[0][0] * SubFactor11 - this->value[0][1] * SubFactor09 + this->value[0][3] * SubFactor12),
            - (this->value[0][0] * SubFactor08 - this->value[0][1] * SubFactor10 + this->value[0][2] * SubFactor12),

            - (this->value[0][1] * SubFactor13 - this->value[0][2] * SubFactor14 + this->value[0][3] * SubFactor15),
            + (this->value[0][0] * SubFactor13 - this->value[0][2] * SubFactor16 + this->value[0][3] * SubFactor17),
            - (this->value[0][0] * SubFactor14 - this->value[0][1] * SubFactor16 + this->value[0][3] * SubFactor18),
            + (this->value[0][0] * SubFactor15 - this->value[0][1] * SubFactor17 + this->value[0][2] * SubFactor18));

        valType Determinant = this->value[0][0] * Inverse[0][0] 
                      + this->value[0][1] * Inverse[1][0] 
                      + this->value[0][2] * Inverse[2][0] 
                      + this->value[0][3] * Inverse[3][0];

        Inverse /= Determinant;
        return Inverse;
    }

	// Binary operators
    template <typename valType> 
    inline tmat4x4<valType> operator+ 
	(
		tmat4x4<valType> const & m, 
		valType const & s
	)
    {
        return tmat4x4<valType>(
            m[0] + s,
            m[1] + s,
            m[2] + s,
            m[3] + s);
    }

    template <typename valType> 
    inline tmat4x4<valType> operator+ 
	(
		valType const & s, 
		tmat4x4<valType> const & m
	)
    {
        return tmat4x4<valType>(
            m[0] + s,
            m[1] + s,
            m[2] + s,
            m[3] + s);
    }

    template <typename valType> 
    inline tmat4x4<valType> operator+ 
	(
		tmat4x4<valType> const & m1, 
		tmat4x4<valType> const & m2
	)
    {
        return tmat4x4<valType>(
            m1[0] + m2[0],
            m1[1] + m2[1],
            m1[2] + m2[2],
            m1[3] + m2[3]);
    }
    
    template <typename valType> 
    inline tmat4x4<valType> operator- 
	(
		tmat4x4<valType> const & m, 
		valType const & s
	)
    {
        return tmat4x4<valType>(
            m[0] - s,
            m[1] - s,
            m[2] - s,
            m[3] - s);
    }

    template <typename valType> 
    inline tmat4x4<valType> operator- 
	(
		valType const & s, 
		tmat4x4<valType> const & m
	)
    {
        return tmat4x4<valType>(
            s - m[0],
            s - m[1],
            s - m[2],
            s - m[3]);
    }

    template <typename valType> 
    inline tmat4x4<valType> operator- 
	(
		tmat4x4<valType> const & m1, 
		tmat4x4<valType> const & m2
	)
    {
        return tmat4x4<valType>(
            m1[0] - m2[0],
            m1[1] - m2[1],
            m1[2] - m2[2],
            m1[3] - m2[3]);
    }

    template <typename valType> 
    inline tmat4x4<valType> operator* 
	(
		tmat4x4<valType> const & m, 
		valType const  & s
	)
    {
        return tmat4x4<valType>(
            m[0] * s,
            m[1] * s,
            m[2] * s,
            m[3] * s);
    }

    template <typename valType> 
    inline tmat4x4<valType> operator* 
	(
		valType const & s, 
		tmat4x4<valType> const & m
	)
    {
        return tmat4x4<valType>(
            m[0] * s,
            m[1] * s,
            m[2] * s,
            m[3] * s);
    }

    template <typename valType> 
    inline detail::tvec4<valType> operator* 
	(
		tmat4x4<valType> const & m, 
		detail::tvec4<valType> const & v
	)
    {
        return detail::tvec4<valType>(
            m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w,
            m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w,
            m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2] * v.w,
            m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z + m[3][3] * v.w);
    }

    template <typename valType> 
    inline detail::tvec4<valType> operator* 
	(
		detail::tvec4<valType> const & v, 
		tmat4x4<valType> const & m
	)
    {
        return detail::tvec4<valType>(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
            m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w);
    }
/*
    template <typename valType> 
    inline tmat4x4<valType> operator* 
	(
		tmat4x4<valType> const & m1, 
		tmat4x4<valType> const & m2
	)
    {
        const valType SrcA00 = m1[0][0];
        const valType SrcA01 = m1[0][1];
        const valType SrcA02 = m1[0][2];
        const valType SrcA03 = m1[0][3];
        const valType SrcA10 = m1[1][0];
        const valType SrcA11 = m1[1][1];
        const valType SrcA12 = m1[1][2];
        const valType SrcA13 = m1[1][3];
        const valType SrcA20 = m1[2][0];
        const valType SrcA21 = m1[2][1];
        const valType SrcA22 = m1[2][2];
        const valType SrcA23 = m1[2][3];
        const valType SrcA30 = m1[3][0];
        const valType SrcA31 = m1[3][1];
        const valType SrcA32 = m1[3][2];
        const valType SrcA33 = m1[3][3];

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
        const valType SrcB30 = m2[3][0];
        const valType SrcB31 = m2[3][1];
        const valType SrcB32 = m2[3][2];
        const valType SrcB33 = m2[3][3];

        tmat4x4<valType> Result;
        Result[0][0] = SrcA00 * SrcB00 + SrcA10 * SrcB01 + SrcA20 * SrcB02 + SrcA30 * SrcB03;
        Result[0][1] = SrcA01 * SrcB00 + SrcA11 * SrcB01 + SrcA21 * SrcB02 + SrcA31 * SrcB03;
        Result[0][2] = SrcA02 * SrcB00 + SrcA12 * SrcB01 + SrcA22 * SrcB02 + SrcA32 * SrcB03;
        Result[0][3] = SrcA03 * SrcB00 + SrcA13 * SrcB01 + SrcA23 * SrcB02 + SrcA33 * SrcB03;
        Result[1][0] = SrcA00 * SrcB10 + SrcA10 * SrcB11 + SrcA20 * SrcB12 + SrcA30 * SrcB13;
        Result[1][1] = SrcA01 * SrcB10 + SrcA11 * SrcB11 + SrcA21 * SrcB12 + SrcA31 * SrcB13;
        Result[1][2] = SrcA02 * SrcB10 + SrcA12 * SrcB11 + SrcA22 * SrcB12 + SrcA32 * SrcB13;
        Result[1][3] = SrcA03 * SrcB10 + SrcA13 * SrcB11 + SrcA23 * SrcB12 + SrcA33 * SrcB13;
        Result[2][0] = SrcA00 * SrcB20 + SrcA10 * SrcB21 + SrcA20 * SrcB22 + SrcA30 * SrcB23;
        Result[2][1] = SrcA01 * SrcB20 + SrcA11 * SrcB21 + SrcA21 * SrcB22 + SrcA31 * SrcB23;
        Result[2][2] = SrcA02 * SrcB20 + SrcA12 * SrcB21 + SrcA22 * SrcB22 + SrcA32 * SrcB23;
        Result[2][3] = SrcA03 * SrcB20 + SrcA13 * SrcB21 + SrcA23 * SrcB22 + SrcA33 * SrcB23;
        Result[3][0] = SrcA00 * SrcB30 + SrcA10 * SrcB31 + SrcA20 * SrcB32 + SrcA30 * SrcB33;
        Result[3][1] = SrcA01 * SrcB30 + SrcA11 * SrcB31 + SrcA21 * SrcB32 + SrcA31 * SrcB33;
        Result[3][2] = SrcA02 * SrcB30 + SrcA12 * SrcB31 + SrcA22 * SrcB32 + SrcA32 * SrcB33;
        Result[3][3] = SrcA03 * SrcB30 + SrcA13 * SrcB31 + SrcA23 * SrcB32 + SrcA33 * SrcB33;
        return Result;
    }
*/
    template <typename valType> 
    inline tmat4x4<valType> operator* 
	(
		tmat4x4<valType> const & m1, 
		tmat4x4<valType> const & m2
	)
    {
		detail::tvec4<valType> const SrcA0 = m1[0];
		detail::tvec4<valType> const SrcA1 = m1[1];
		detail::tvec4<valType> const SrcA2 = m1[2];
		detail::tvec4<valType> const SrcA3 = m1[3];

		detail::tvec4<valType> const SrcB0 = m2[0];
		detail::tvec4<valType> const SrcB1 = m2[1];
		detail::tvec4<valType> const SrcB2 = m2[2];
		detail::tvec4<valType> const SrcB3 = m2[3];

        tmat4x4<valType> Result;
		Result[0] = SrcA0 * SrcB0[0] + SrcA1 * SrcB0[1] + SrcA2 * SrcB0[2] + SrcA3 * SrcB0[3];
		Result[1] = SrcA0 * SrcB1[0] + SrcA1 * SrcB1[1] + SrcA2 * SrcB1[2] + SrcA3 * SrcB1[3];
		Result[2] = SrcA0 * SrcB2[0] + SrcA1 * SrcB2[1] + SrcA2 * SrcB2[2] + SrcA3 * SrcB2[3];
		Result[3] = SrcA0 * SrcB3[0] + SrcA1 * SrcB3[1] + SrcA2 * SrcB3[2] + SrcA3 * SrcB3[3];
        return Result;
    }

    template <typename valType> 
    inline tmat4x4<valType> operator/ 
	(
		tmat4x4<valType> const & m, 
		valType const & s
	)
    {
        return tmat4x4<valType>(
            m[0] / s,
            m[1] / s,
            m[2] / s,
            m[3] / s);
    }

    template <typename valType> 
    inline tmat4x4<valType> operator/ (const valType s, const tmat4x4<valType>& m)
    {
        return tmat4x4<valType>(
            s / m[0],
            s / m[1],
            s / m[2],
            s / m[3]);
    }

    template <typename valType> 
    inline tvec4<valType> operator/ 
	(
		tmat4x4<valType> const & m, 
		tvec4<valType> const & v
	)
    {
        return m._inverse() * v;
    }

    template <typename valType> 
    inline tvec4<valType> operator/ 
	(
		tvec4<valType> const & v, 
		tmat4x4<valType> const & m
	)
    {
        return v * m._inverse();
    }
 
    template <typename valType> 
    inline tmat4x4<valType> operator/ 
	(
		tmat4x4<valType> const & m1, 
		tmat4x4<valType> const & m2
	)
    {
        return m1 * m2._inverse();
    }

	// Unary constant operators
    template <typename valType> 
    inline tmat4x4<valType> const operator- 
	(
		tmat4x4<valType> const & m
	)
    {
        return tmat4x4<valType>(
            -m[0], 
            -m[1],
            -m[2],
            -m[3]);
    }

    template <typename valType> 
    inline tmat4x4<valType> const operator++ 
	(
		tmat4x4<valType> const & m, 
		int
	) 
    {
        return tmat4x4<valType>(
            m[0] + valType(1),
            m[1] + valType(1),
            m[2] + valType(1),
            m[3] + valType(1));
    }

    template <typename valType> 
    inline tmat4x4<valType> const operator-- 
	(
		tmat4x4<valType> const & m, 
		int
	) 
    {
        return tmat4x4<valType>(
            m[0] - valType(1),
            m[1] - valType(1),
            m[2] - valType(1),
            m[3] - valType(1));
    }

} //namespace detail
} //namespace glm
