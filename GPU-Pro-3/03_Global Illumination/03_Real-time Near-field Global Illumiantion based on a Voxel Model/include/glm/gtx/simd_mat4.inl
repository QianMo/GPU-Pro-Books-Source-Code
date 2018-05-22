///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-05-19
// Updated : 2009-05-19
// Licence : This source is under MIT License
// File    : glm/gtx/simd_mat4.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail
{
    inline fmat4x4SIMD::fmat4x4SIMD()
    {}

    inline fmat4x4SIMD::fmat4x4SIMD(float const & s)
    {
        this->value[0] = fvec4SIMD(s, 0, 0, 0);
        this->value[1] = fvec4SIMD(0, s, 0, 0);
        this->value[2] = fvec4SIMD(0, 0, s, 0);
        this->value[3] = fvec4SIMD(0, 0, 0, s);
    }

	inline fmat4x4SIMD::fmat4x4SIMD
	(
		float const & x0, float const & y0, float const & z0, float const & w0,
		float const & x1, float const & y1, float const & z1, float const & w1,
		float const & x2, float const & y2, float const & z2, float const & w2,
		float const & x3, float const & y3, float const & z3, float const & w3
	)
	{
        this->value[0] = fvec4SIMD(x0, y0, z0, w0);
        this->value[1] = fvec4SIMD(x1, y1, z1, w1);
        this->value[2] = fvec4SIMD(x2, y2, z2, w2);
        this->value[3] = fvec4SIMD(x3, y3, z3, w3);
	}

	inline fmat4x4SIMD::fmat4x4SIMD
	(
		fvec4SIMD const & v0,
		fvec4SIMD const & v1,
		fvec4SIMD const & v2,
		fvec4SIMD const & v3
	)
	{
        this->value[0] = v0;
        this->value[1] = v1;
        this->value[2] = v2;
        this->value[3] = v3;
	}

	inline fmat4x4SIMD::fmat4x4SIMD
	(
		tmat4x4 const & m
	)
	{
        this->value[0] = fvec4SIMD(m[0]);
        this->value[1] = fvec4SIMD(m[1]);
        this->value[2] = fvec4SIMD(m[2]);
        this->value[3] = fvec4SIMD(m[3]);
	}

	//////////////////////////////////////
	// Accesses

	inline fvec4SIMD & fmat4x4SIMD::operator[]
	(
		typename fmat4x4SIMD::size_type i
	)
	{
		assert(
			i >= typename tmat4x4<valType>::size_type(0) && 
			i < tmat4x4<valType>::col_size());

		return value[i];
	}

	inline fvec4SIMD const & fmat4x4SIMD::operator[]
	(
		typename fmat4x4SIMD::size_type i
	) const
	{
		assert(
			i >= typename fmat4x4SIMD::size_type(0) && 
			i < fmat4x4SIMD::col_size());

		return value[i];
	}

    //////////////////////////////////////////////////////////////
    // mat4 operators

    inline fmat4x4SIMD& fmat4x4SIMD::operator= 
	(
		fmat4x4SIMD const & m
	)
    {
        this->value[0].Data = m[0].Data;
        this->value[1].Data = m[1].Data;
        this->value[2].Data = m[2].Data;
        this->value[3].Data = m[3].Data;
        return *this;
    }

    inline fmat4x4SIMD & fmat4x4SIMD::operator+= 
	(
		fmat4x4SIMD const & m
	)
    {
		this->value[0].Data = _mm_add_ps(this->value[0].Data, m[0].Data);
        this->value[1].Data = _mm_add_ps(this->value[1].Data, m[1].Data);
        this->value[2].Data = _mm_add_ps(this->value[2].Data, m[2].Data);
        this->value[3].Data = _mm_add_ps(this->value[3].Data, m[3].Data);
        return *this;
    }

    inline fmat4x4SIMD & fmat4x4SIMD::operator-= 
	(
		fmat4x4SIMD const & m
	)
    {
		this->value[0].Data = _mm_sub_ps(this->value[0].Data, m[0].Data);
        this->value[1].Data = _mm_sub_ps(this->value[1].Data, m[1].Data);
        this->value[2].Data = _mm_sub_ps(this->value[2].Data, m[2].Data);
        this->value[3].Data = _mm_sub_ps(this->value[3].Data, m[3].Data);

        return *this;
    }

    inline fmat4x4SIMD & fmat4x4SIMD::operator*= 
	(
		fmat4x4SIMD const & m
	)
    {
		_mm_mul_ps(this->Data, m.Data, this->Data);
        return *this;
    }

    inline fmat4x4SIMD & fmat4x4SIMD::operator/= 
	(
		fmat4x4SIMD const & m
	)
    {
		__m128 Inv[4];
		_mm_inverse_ps(m.Data, Inv);
		_mm_mul_ps(this->Data, Inv, this->Data);
        return *this;
    }

    inline fmat4x4SIMD & fmat4x4SIMD::operator+= 
	(
		float const & s
	)
    {
		__m128 Operand = _mm_set_ps1(s);
		this->value[0].Data = _mm_add_ps(this->value[0].Data, Operand);
        this->value[1].Data = _mm_add_ps(this->value[1].Data, Operand);
        this->value[2].Data = _mm_add_ps(this->value[2].Data, Operand);
        this->value[3].Data = _mm_add_ps(this->value[3].Data, Operand);
        return *this;
    }

    inline fmat4x4SIMD & fmat4x4SIMD::operator-= 
	(
		float const & s
	)
    {
		__m128 Operand = _mm_set_ps1(s);
        this->value[0].Data = _mm_sub_ps(this->value[0].Data, Operand);
        this->value[1].Data = _mm_sub_ps(this->value[1].Data, Operand);
        this->value[2].Data = _mm_sub_ps(this->value[2].Data, Operand);
        this->value[3].Data = _mm_sub_ps(this->value[3].Data, Operand);
        return *this;
    }

    inline fmat4x4SIMD & fmat4x4SIMD::operator*= 
	(
		float const & s
	)
    {
		__m128 Operand = _mm_set_ps1(s);
        this->value[0].Data = _mm_mul_ps(this->value[0].Data, Operand);
        this->value[1].Data = _mm_mul_ps(this->value[1].Data, Operand);
        this->value[2].Data = _mm_mul_ps(this->value[2].Data, Operand);
        this->value[3].Data = _mm_mul_ps(this->value[3].Data, Operand);
        return *this;
    }

    inline fmat4x4SIMD & fmat4x4SIMD::operator/= 
	(
		float const & s
	)
    {
		__m128 Operand = _mm_div_ps(one, s));
        this->value[0].Data = _mm_mul_ps(this->value[0].Data, Operand);
        this->value[1].Data = _mm_mul_ps(this->value[1].Data, Operand);
        this->value[2].Data = _mm_mul_ps(this->value[2].Data, Operand);
        this->value[3].Data = _mm_mul_ps(this->value[3].Data, Operand);
        return *this;
    }

    inline fmat4x4SIMD & fmat4x4SIMD::operator++ ()
    {
		this->value[0].Data = _mm_add_ps(this->value[0].Data, one);
        this->value[1].Data = _mm_add_ps(this->value[1].Data, one);
        this->value[2].Data = _mm_add_ps(this->value[2].Data, one);
        this->value[3].Data = _mm_add_ps(this->value[3].Data, one);
        return *this;
    }

    inline fmat4x4SIMD & fmat4x4SIMD::operator-- ()
    {
		this->value[0].Data = _mm_sub_ps(this->value[0].Data, one);
        this->value[1].Data = _mm_sub_ps(this->value[1].Data, one);
        this->value[2].Data = _mm_sub_ps(this->value[2].Data, one);
        this->value[3].Data = _mm_sub_ps(this->value[3].Data, one);
        return *this;
    }

}//namespace detail
}//namespace glm
