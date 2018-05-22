///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-03-08
// Updated : 2008-03-08
// Licence : This source is under MIT License
// File    : glm/core/func_matrix.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	namespace core{
	namespace function{
	namespace matrix{

    // matrixCompMult
/*
    template <typename valType>
    inline detail::tmat2x2<valType> matrixCompMult
	(
		detail::tmat2x2<valType> const & x, 
		detail::tmat2x2<valType> const & y
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat2x2<valType> result;
		for(std::size_t i = 0; i < detail::tmat2x2<valType>::col_size(); ++i)
			result[i] = x[i] * y[i];
        return result;
    }

    template <typename valType>
	inline detail::tmat3x3<valType> matrixCompMult
	(
		detail::tmat3x3<valType> const & x, 
		detail::tmat3x3<valType> const & y
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat3x3<valType> result;
		for(std::size_t i = 0; i < detail::tmat3x3<valType>::col_size(); ++i)
			result[i] = x[i] * y[i];
        return result;
    }

    template <typename valType>
    inline detail::tmat4x4<valType> matrixCompMult
	(
		detail::tmat4x4<valType> const & x, 
		detail::tmat4x4<valType> const & y
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat4x4<valType> result;
		for(std::size_t i = 0; i < detail::tmat4x4<valType>::col_size(); ++i)
			result[i] = x[i] * y[i];
        return result;
    }

    template <typename valType>
    inline detail::tmat2x3<valType> matrixCompMult
	(
		detail::tmat2x3<valType> const & x, 
		detail::tmat2x3<valType> const & y
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat2x3<valType> result;
		for(std::size_t i = 0; i < detail::tmat2x3<valType>::col_size(); ++i)
			result[i] = x[i] * y[i];
        return result;
    }

    template <typename valType>
    inline detail::tmat3x2<valType> matrixCompMult
	(
		detail::tmat3x2<valType> const & x, 
		detail::tmat3x2<valType> const & y
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat3x2<valType> result;
		for(std::size_t i = 0; i < detail::tmat3x2<valType>::col_size(); ++i)
			result[i] = x[i] * y[i];
        return result;
    }

    template <typename valType>
    inline detail::tmat2x4<valType> matrixCompMult
	(
		detail::tmat2x4<valType> const & x, 
		detail::tmat2x4<valType> const & y
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat2x4<valType> result;
		for(std::size_t i = 0; i < detail::tmat2x4<valType>::col_size(); ++i)
			result[i] = x[i] * y[i];
        return result;
    }

    template <typename valType>
    inline detail::tmat4x2<valType> matrixCompMult
	(
		detail::tmat4x2<valType> const & x, 
		detail::tmat4x2<valType> const & y
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat4x2<valType> result;
		for(std::size_t i = 0; i < detail::tmat4x2<valType>::col_size(); ++i)
			result[i] = x[i] * y[i];
        return result;
    }

    template <typename valType>
    inline detail::tmat3x4<valType> matrixCompMult
	(
		detail::tmat3x4<valType> const & x, 
		detail::tmat3x4<valType> const & y
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat3x4<valType> result;
		for(std::size_t i = 0; i < detail::tmat3x4<valType>::col_size(); ++i)
			result[i] = x[i] * y[i];
        return result;
    }

    template <typename valType>
    inline detail::tmat4x3<valType> matrixCompMult
	(
		detail::tmat4x3<valType> const & x, 
		detail::tmat4x3<valType> const & y
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat4x3<valType> result;
		for(std::size_t i = 0; i < detail::tmat4x3<valType>::col_size(); ++i)
			result[i] = x[i] * y[i];
        return result;
    }
*/
    template <typename matType>
    inline matType matrixCompMult
	(
		matType const & x, 
		matType const & y
	)
    {
		GLM_STATIC_ASSERT(detail::type<typename matType::value_type>::is_float);

        matType result;
		for(typename matType::size_type i = 0; i < matType::col_size(); ++i)
			result[i] = x[i] * y[i];
        return result;
    }

	// outerProduct
    template <typename valType>
    inline detail::tmat2x2<valType> outerProduct
	(
		detail::tvec2<valType> const & c, 
		detail::tvec2<valType> const & r
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		detail::tmat2x2<valType> m;
		m[0][0] = c[0] * r[0];
		m[0][1] = c[1] * r[0];
		m[1][0] = c[0] * r[1];
		m[1][1] = c[1] * r[1];
        return m;
    }

    template <typename valType>
    inline detail::tmat3x3<valType> outerProduct
	(
		detail::tvec3<valType> const & c, 
		detail::tvec3<valType> const & r
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		detail::tmat3x3<valType> m;
		for(typename detail::tmat3x3<valType>::size_type i = 0; i < detail::tmat3x3<valType>::col_size(); ++i)
			m[i] = c * r[i];
        return m;
    }

    template <typename valType>
    inline detail::tmat4x4<valType> outerProduct
	(
		detail::tvec4<valType> const & c, 
		detail::tvec4<valType> const & r
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		detail::tmat4x4<valType> m;
		for(typename detail::tmat4x4<valType>::size_type i = 0; i < detail::tmat4x4<valType>::col_size(); ++i)
			m[i] = c * r[i];
        return m;
    }

    template <typename valType>
	inline detail::tmat2x3<valType> outerProduct
	(
		detail::tvec3<valType> const & c, 
		detail::tvec2<valType> const & r
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		detail::tmat2x3<valType> m;
		m[0][0] = c.x * r.x;
		m[0][1] = c.y * r.x;
		m[0][2] = c.z * r.x;
		m[1][0] = c.x * r.y;
		m[1][1] = c.y * r.y;
		m[1][2] = c.z * r.y;
		return m;
	}

    template <typename valType>
	inline detail::tmat3x2<valType> outerProduct
	(
		detail::tvec2<valType> const & c, 
		detail::tvec3<valType> const & r
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		detail::tmat3x2<valType> m;
		m[0][0] = c.x * r.x;
		m[0][1] = c.y * r.x;
		m[1][0] = c.x * r.y;
		m[1][1] = c.y * r.y;
		m[2][0] = c.x * r.z;
		m[2][1] = c.y * r.z;
		return m;
	}

	template <typename valType>
	inline detail::tmat2x4<valType> outerProduct
	(
		detail::tvec2<valType> const & c, 
		detail::tvec4<valType> const & r
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		detail::tmat2x4<valType> m;
		m[0][0] = c.x * r.x;
		m[0][1] = c.y * r.x;
		m[0][2] = c.z * r.x;
		m[0][3] = c.w * r.x;
		m[1][0] = c.x * r.y;
		m[1][1] = c.y * r.y;
		m[1][2] = c.z * r.y;
		m[1][3] = c.w * r.y;
		return m;
	}

	template <typename valType>
	inline detail::tmat4x2<valType> outerProduct
	(
		detail::tvec4<valType> const & c, 
		detail::tvec2<valType> const & r
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		detail::tmat4x2<valType> m;
		m[0][0] = c.x * r.x;
		m[0][1] = c.y * r.x;
		m[1][0] = c.x * r.y;
		m[1][1] = c.y * r.y;
		m[2][0] = c.x * r.z;
		m[2][1] = c.y * r.z;
		m[3][0] = c.x * r.w;
		m[3][1] = c.y * r.w;
		return m;
	}

	template <typename valType>
	inline detail::tmat3x4<valType> outerProduct
	(
		detail::tvec4<valType> const & c, 
		detail::tvec3<valType> const & r
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		detail::tmat3x4<valType> m;
		m[0][0] = c.x * r.x;
		m[0][1] = c.y * r.x;
		m[0][2] = c.z * r.x;
		m[0][3] = c.w * r.x;
		m[1][0] = c.x * r.y;
		m[1][1] = c.y * r.y;
		m[1][2] = c.z * r.y;
		m[1][3] = c.w * r.y;
		m[2][0] = c.x * r.z;
		m[2][1] = c.y * r.z;
		m[2][2] = c.z * r.z;
		m[2][3] = c.w * r.z;
		return m;
	}

	template <typename valType>
	inline detail::tmat4x3<valType> outerProduct
	(
		detail::tvec3<valType> const & c, 
		detail::tvec4<valType> const & r
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		detail::tmat4x3<valType> m;
		m[0][0] = c.x * r.x;
		m[0][1] = c.y * r.x;
		m[0][2] = c.z * r.x;
		m[1][0] = c.x * r.y;
		m[1][1] = c.y * r.y;
		m[1][2] = c.z * r.y;
		m[2][0] = c.x * r.z;
		m[2][1] = c.y * r.z;
		m[2][2] = c.z * r.z;
		m[3][0] = c.x * r.w;
		m[3][1] = c.y * r.w;
		m[3][2] = c.z * r.w;
		return m;
	}

    template <typename valType>
    inline detail::tmat2x2<valType> transpose
	(
		detail::tmat2x2<valType> const & m
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat2x2<valType> result;
        result[0][0] = m[0][0];
        result[0][1] = m[1][0];
        result[1][0] = m[0][1];
        result[1][1] = m[1][1];
        return result;
    }

    template <typename valType>
    inline detail::tmat3x3<valType> transpose
	(
		detail::tmat3x3<valType> const & m
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat3x3<valType> result;
        result[0][0] = m[0][0];
        result[0][1] = m[1][0];
        result[0][2] = m[2][0];

        result[1][0] = m[0][1];
        result[1][1] = m[1][1];
        result[1][2] = m[2][1];

        result[2][0] = m[0][2];
        result[2][1] = m[1][2];
        result[2][2] = m[2][2];
        return result;
    }

    template <typename valType>
    inline detail::tmat4x4<valType> transpose
	(
		detail::tmat4x4<valType> const & m
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat4x4<valType> result;
        result[0][0] = m[0][0];
        result[0][1] = m[1][0];
        result[0][2] = m[2][0];
        result[0][3] = m[3][0];

        result[1][0] = m[0][1];
        result[1][1] = m[1][1];
        result[1][2] = m[2][1];
        result[1][3] = m[3][1];

        result[2][0] = m[0][2];
        result[2][1] = m[1][2];
        result[2][2] = m[2][2];
        result[2][3] = m[3][2];

        result[3][0] = m[0][3];
        result[3][1] = m[1][3];
        result[3][2] = m[2][3];
        result[3][3] = m[3][3];
        return result;
    }

    template <typename valType>
    inline detail::tmat2x3<valType> transpose
	(
		detail::tmat3x2<valType> const & m
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat2x3<valType> result;
        result[0][0] = m[0][0];
        result[0][1] = m[1][0];
		result[0][2] = m[2][0];
        result[1][0] = m[0][1];
        result[1][1] = m[1][1];
		result[1][2] = m[2][1];
        return result;
    }

    template <typename valType>
    inline detail::tmat3x2<valType> transpose
	(
		detail::tmat2x3<valType> const & m
	)
    {
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat3x2<valType> result;
        result[0][0] = m[0][0];
        result[0][1] = m[1][0];
        result[1][0] = m[0][1];
        result[1][1] = m[1][1];
        result[2][0] = m[0][2];
        result[2][1] = m[1][2];
        return result;
    }

    template <typename valType>
	inline detail::tmat2x4<valType> transpose
	(
		detail::tmat4x2<valType> const & m
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		detail::tmat2x4<valType> result;
        result[0][0] = m[0][0];
        result[0][1] = m[1][0];
		result[0][2] = m[2][0];
		result[0][3] = m[3][0];
        result[1][0] = m[0][1];
        result[1][1] = m[1][1];
		result[1][2] = m[2][1];
		result[1][3] = m[3][1];
		return result;
	}

    template <typename valType>
	inline detail::tmat4x2<valType> transpose
	(
		detail::tmat2x4<valType> const & m
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat4x2<valType> result;
        result[0][0] = m[0][0];
        result[0][1] = m[1][0];
        result[1][0] = m[0][1];
        result[1][1] = m[1][1];
        result[2][0] = m[0][2];
        result[2][1] = m[1][2];
        result[3][0] = m[0][3];
        result[3][1] = m[1][3];
        return result;
	}

    template <typename valType>
	inline detail::tmat3x4<valType> transpose
	(
		detail::tmat4x3<valType> const & m
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		detail::tmat3x4<valType> result;
        result[0][0] = m[0][0];
        result[0][1] = m[1][0];
		result[0][2] = m[2][0];
		result[0][3] = m[3][0];
        result[1][0] = m[0][1];
        result[1][1] = m[1][1];
		result[1][2] = m[2][1];
		result[1][3] = m[3][1];
        result[2][0] = m[0][2];
        result[2][1] = m[1][2];
		result[2][2] = m[2][2];
		result[2][3] = m[3][2];
		return result;
	}

    template <typename valType>
	inline detail::tmat4x3<valType> transpose
	(
		detail::tmat3x4<valType> const & m
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

        detail::tmat4x3<valType> result;
        result[0][0] = m[0][0];
        result[0][1] = m[1][0];
		result[0][2] = m[2][0];
        result[1][0] = m[0][1];
        result[1][1] = m[1][1];
		result[1][2] = m[2][1];
        result[2][0] = m[0][2];
        result[2][1] = m[1][2];
		result[2][2] = m[2][2];
        result[3][0] = m[0][3];
        result[3][1] = m[1][3];
		result[3][2] = m[2][3];
        return result;
	}

	template <typename valType>
	inline typename detail::tmat2x2<valType>::value_type determinant
	(
		detail::tmat2x2<valType> const & m
	)
	{
		return m[0][0] * m[1][1] - m[1][0] * m[0][1];
	}

	template <typename valType>
	inline typename detail::tmat3x3<valType>::value_type determinant
	(
		detail::tmat3x3<valType> const & m
	)
	{
		return m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
			- m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
			+ m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
	}

	template <typename valType>
	inline typename detail::tmat4x4<valType>::value_type determinant
	(
		detail::tmat4x4<valType> const & m
	)
	{
		valType SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		valType SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		valType SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		valType SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		valType SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		valType SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

		detail::tvec4<valType> DetCof(
			+ (m[1][1] * SubFactor00 - m[1][2] * SubFactor01 + m[1][3] * SubFactor02),
			- (m[1][0] * SubFactor00 - m[1][2] * SubFactor03 + m[1][3] * SubFactor04),
			+ (m[1][0] * SubFactor01 - m[1][1] * SubFactor03 + m[1][3] * SubFactor05),
			- (m[1][0] * SubFactor02 - m[1][1] * SubFactor04 + m[1][2] * SubFactor05));

		return m[0][0] * DetCof[0]
			 + m[0][1] * DetCof[1]
			 + m[0][2] * DetCof[2]
			 + m[0][3] * DetCof[3];
	}

	template <typename valType> 
	inline detail::tmat2x2<valType> inverse
	(
		detail::tmat2x2<valType> const & m
	)
	{
		//valType Determinant = m[0][0] * m[1][1] - m[1][0] * m[0][1];
		valType Determinant = determinant(m);

		detail::tmat2x2<valType> Inverse(
			+ m[1][1] / Determinant,
			- m[1][0] / Determinant,
			- m[0][1] / Determinant, 
			+ m[0][0] / Determinant);

		return Inverse;
	}

	template <typename valType> 
	inline detail::tmat3x3<valType> inverse
	(
		detail::tmat3x3<valType> const & m
	)
	{
		//valType Determinant = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
		//					- m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
		//					+ m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);

		valType Determinant = determinant(m);

		detail::tmat3x3<valType> Inverse;
		Inverse[0][0] = + (m[1][1] * m[2][2] - m[2][1] * m[1][2]);
		Inverse[1][0] = - (m[1][0] * m[2][2] - m[2][0] * m[1][2]);
		Inverse[2][0] = + (m[1][0] * m[2][1] - m[2][0] * m[1][1]);
		Inverse[0][1] = - (m[0][1] * m[2][2] - m[2][1] * m[0][2]);
		Inverse[1][1] = + (m[0][0] * m[2][2] - m[2][0] * m[0][2]);
		Inverse[2][1] = - (m[0][0] * m[2][1] - m[2][0] * m[0][1]);
		Inverse[0][2] = + (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
		Inverse[1][2] = - (m[0][0] * m[1][2] - m[1][0] * m[0][2]);
		Inverse[2][2] = + (m[0][0] * m[1][1] - m[1][0] * m[0][1]);
		Inverse /= Determinant;

		return Inverse;
	}

    template <typename valType> 
    inline detail::tmat4x4<valType> inverseOgre
	(
		detail::tmat4x4<valType> const & m
	)
    {
        valType m00 = m[0][0], m01 = m[0][1], m02 = m[0][2], m03 = m[0][3];
        valType m10 = m[1][0], m11 = m[1][1], m12 = m[1][2], m13 = m[1][3];
        valType m20 = m[2][0], m21 = m[2][1], m22 = m[2][2], m23 = m[2][3];
        valType m30 = m[3][0], m31 = m[3][1], m32 = m[3][2], m33 = m[3][3];

        valType v0 = m20 * m31 - m21 * m30;
        valType v1 = m20 * m32 - m22 * m30;
        valType v2 = m20 * m33 - m23 * m30;
        valType v3 = m21 * m32 - m22 * m31;
        valType v4 = m21 * m33 - m23 * m31;
        valType v5 = m22 * m33 - m23 * m32;

        valType t00 = + (v5 * m11 - v4 * m12 + v3 * m13);
        valType t10 = - (v5 * m10 - v2 * m12 + v1 * m13);
        valType t20 = + (v4 * m10 - v2 * m11 + v0 * m13);
        valType t30 = - (v3 * m10 - v1 * m11 + v0 * m12);

        valType invDet = 1 / (t00 * m00 + t10 * m01 + t20 * m02 + t30 * m03);

        valType d00 = t00 * invDet;
        valType d10 = t10 * invDet;
        valType d20 = t20 * invDet;
        valType d30 = t30 * invDet;

        valType d01 = - (v5 * m01 - v4 * m02 + v3 * m03) * invDet;
        valType d11 = + (v5 * m00 - v2 * m02 + v1 * m03) * invDet;
        valType d21 = - (v4 * m00 - v2 * m01 + v0 * m03) * invDet;
        valType d31 = + (v3 * m00 - v1 * m01 + v0 * m02) * invDet;

        v0 = m10 * m31 - m11 * m30;
        v1 = m10 * m32 - m12 * m30;
        v2 = m10 * m33 - m13 * m30;
        v3 = m11 * m32 - m12 * m31;
        v4 = m11 * m33 - m13 * m31;
        v5 = m12 * m33 - m13 * m32;

        valType d02 = + (v5 * m01 - v4 * m02 + v3 * m03) * invDet;
        valType d12 = - (v5 * m00 - v2 * m02 + v1 * m03) * invDet;
        valType d22 = + (v4 * m00 - v2 * m01 + v0 * m03) * invDet;
        valType d32 = - (v3 * m00 - v1 * m01 + v0 * m02) * invDet;

        v0 = m21 * m10 - m20 * m11;
        v1 = m22 * m10 - m20 * m12;
        v2 = m23 * m10 - m20 * m13;
        v3 = m22 * m11 - m21 * m12;
        v4 = m23 * m11 - m21 * m13;
        v5 = m23 * m12 - m22 * m13;

        valType d03 = - (v5 * m01 - v4 * m02 + v3 * m03) * invDet;
        valType d13 = + (v5 * m00 - v2 * m02 + v1 * m03) * invDet;
        valType d23 = - (v4 * m00 - v2 * m01 + v0 * m03) * invDet;
        valType d33 = + (v3 * m00 - v1 * m01 + v0 * m02) * invDet;

        return detail::tmat4x4<valType>(
            d00, d01, d02, d03,
            d10, d11, d12, d13,
            d20, d21, d22, d23,
            d30, d31, d32, d33);
    }

	template <typename valType> 
	inline detail::tmat4x4<valType> inverse
	(
		detail::tmat4x4<valType> const & m
	)
	{
		valType Coef00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		valType Coef02 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
		valType Coef03 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

		valType Coef04 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		valType Coef06 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
		valType Coef07 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

		valType Coef08 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		valType Coef10 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
		valType Coef11 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

		valType Coef12 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		valType Coef14 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
		valType Coef15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

		valType Coef16 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		valType Coef18 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
		valType Coef19 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

		valType Coef20 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
		valType Coef22 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
		valType Coef23 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

		detail::tvec4<valType> const SignA(+1, -1, +1, -1);
		detail::tvec4<valType> const SignB(-1, +1, -1, +1);

		detail::tvec4<valType> Fac0(Coef00, Coef00, Coef02, Coef03);
		detail::tvec4<valType> Fac1(Coef04, Coef04, Coef06, Coef07);
		detail::tvec4<valType> Fac2(Coef08, Coef08, Coef10, Coef11);
		detail::tvec4<valType> Fac3(Coef12, Coef12, Coef14, Coef15);
		detail::tvec4<valType> Fac4(Coef16, Coef16, Coef18, Coef19);
		detail::tvec4<valType> Fac5(Coef20, Coef20, Coef22, Coef23);

		detail::tvec4<valType> Vec0(m[1][0], m[0][0], m[0][0], m[0][0]);
		detail::tvec4<valType> Vec1(m[1][1], m[0][1], m[0][1], m[0][1]);
		detail::tvec4<valType> Vec2(m[1][2], m[0][2], m[0][2], m[0][2]);
		detail::tvec4<valType> Vec3(m[1][3], m[0][3], m[0][3], m[0][3]);

		detail::tvec4<valType> Inv0 = SignA * (Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2);
		detail::tvec4<valType> Inv1 = SignB * (Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4);
		detail::tvec4<valType> Inv2 = SignA * (Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5);
		detail::tvec4<valType> Inv3 = SignB * (Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5);

		detail::tmat4x4<valType> Inverse(Inv0, Inv1, Inv2, Inv3);

		detail::tvec4<valType> Row0(Inverse[0][0], Inverse[1][0], Inverse[2][0], Inverse[3][0]);

		valType Determinant = glm::dot(m[0], Row0);

		Inverse /= Determinant;
	    
		return Inverse;
	}

	}//namespace matrix
	}//namespace function
	}//namespace core
}//namespace glm
