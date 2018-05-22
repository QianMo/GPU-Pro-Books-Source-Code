///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-08-03
// Updated : 2008-09-14
// Licence : This source is under MIT License
// File    : glm/core/func_trigonometric.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	namespace core{
	namespace function{
	namespace trigonometric{

    // radians
    template <typename genType>
    inline genType radians
	(
		genType const & degrees
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        const genType pi = genType(3.1415926535897932384626433832795);
        return degrees * (pi / genType(180));
    }

    template <typename valType>
	inline detail::tvec2<valType> radians
	(
		detail::tvec2<valType> const & degrees
	)
    {
        return detail::tvec2<valType>(
            radians(degrees.x),
            radians(degrees.y));
    }

    template <typename valType>
    inline detail::tvec3<valType> radians
	(
		detail::tvec3<valType> const & degrees
	)
    {
        return detail::tvec3<valType>(
            radians(degrees.x),
            radians(degrees.y),
            radians(degrees.z));
    }

    template <typename valType>
    inline detail::tvec4<valType> radians
	(
		detail::tvec4<valType> const & degrees
	)
    {
        return detail::tvec4<valType>(
            radians(degrees.x),
            radians(degrees.y),
            radians(degrees.z),
            radians(degrees.w));
    }

    // degrees
    template <typename genType>
    inline genType degrees
	(
		genType const & radians
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        const genType pi = genType(3.1415926535897932384626433832795);
        return radians * (genType(180) / pi);
    }

    template <typename valType>
    inline detail::tvec2<valType> degrees
	(
		detail::tvec2<valType> const & radians
	)
    {
        return detail::tvec2<valType>(
            degrees(radians.x),
            degrees(radians.y));
    }

    template <typename valType>
    inline detail::tvec3<valType> degrees
	(	
		detail::tvec3<valType> const & radians
	)
    {
        return detail::tvec3<valType>(
            degrees(radians.x),
            degrees(radians.y),
            degrees(radians.z));
    }

    template <typename valType>
    inline detail::tvec4<valType> degrees
	(
		detail::tvec4<valType> const & radians
	)
    {
        return detail::tvec4<valType>(
            degrees(radians.x),
            degrees(radians.y),
            degrees(radians.z),
            degrees(radians.w));
    }

    // sin
    template <typename genType>
    inline genType sin
	(
		genType const & angle
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

		return ::std::sin(angle);
    }

    template <typename valType>
    inline detail::tvec2<valType> sin
	(
		detail::tvec2<valType> const & angle
	)
    {
        return detail::tvec2<valType>(
            sin(angle.x),
            sin(angle.y));
    }

    template <typename valType>
    inline detail::tvec3<valType> sin
	(
		detail::tvec3<valType> const & angle
	)
    {
        return detail::tvec3<valType>(
            sin(angle.x),
            sin(angle.y),
            sin(angle.z));
    }

    template <typename valType>
    inline detail::tvec4<valType> sin
	(
		detail::tvec4<valType> const & angle
	)
    {
        return detail::tvec4<valType>(
            sin(angle.x),
            sin(angle.y),
            sin(angle.z),
            sin(angle.w));
    }

    // cos
    template <typename genType>
    inline genType cos(genType const & angle)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        return ::std::cos(angle);
    }

    template <typename valType>
    inline detail::tvec2<valType> cos
	(
		detail::tvec2<valType> const & angle
	)
    {
        return detail::tvec2<valType>(
            cos(angle.x),
            cos(angle.y));
    }

    template <typename valType>
    inline detail::tvec3<valType> cos
	(
		detail::tvec3<valType> const & angle
	)
    {
        return detail::tvec3<valType>(
            cos(angle.x),
            cos(angle.y),
            cos(angle.z));
    }

    template <typename valType>
    inline detail::tvec4<valType> cos
	(	
		detail::tvec4<valType> const & angle
	)
    {
        return detail::tvec4<valType>(
            cos(angle.x),
            cos(angle.y),
            cos(angle.z),
            cos(angle.w));
    }

    // tan
    template <typename genType>
    inline genType tan
	(
		genType const & angle
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        return ::std::tan(angle);
    }

    template <typename valType>
    inline detail::tvec2<valType> tan
	(
		detail::tvec2<valType> const & angle
	)
    {
        return detail::tvec2<valType>(
            tan(angle.x),
            tan(angle.y));
    }

    template <typename valType>
    inline detail::tvec3<valType> tan
	(
		detail::tvec3<valType> const & angle
	)
    {
        return detail::tvec3<valType>(
            tan(angle.x),
            tan(angle.y),
            tan(angle.z));
    }

    template <typename valType>
    inline detail::tvec4<valType> tan
	(
		detail::tvec4<valType> const & angle
	)
    {
        return detail::tvec4<valType>(
            tan(angle.x),
            tan(angle.y),
            tan(angle.z),
            tan(angle.w));
    }

    // asin
    template <typename genType>
    inline genType asin
	(
		genType const & x
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        return ::std::asin(x);
    }

    template <typename valType>
	inline detail::tvec2<valType> asin
	(
		detail::tvec2<valType> const & x
	)
    {
        return detail::tvec2<valType>(
            asin(x.x),
            asin(x.y));
    }

    template <typename valType>
	inline detail::tvec3<valType> asin
	(
		detail::tvec3<valType> const & x
	)
    {
        return detail::tvec3<valType>(
            asin(x.x),
            asin(x.y),
            asin(x.z));
    }

    template <typename valType>
	inline detail::tvec4<valType> asin
	(
		detail::tvec4<valType> const & x
	)
    {
        return detail::tvec4<valType>(
            asin(x.x),
            asin(x.y),
            asin(x.z),
            asin(x.w));
    }

    // acos
    template <typename genType>
    inline genType acos
	(
		genType const & x
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        return ::std::acos(x);
    }

    template <typename valType>
	inline detail::tvec2<valType> acos
	(
		detail::tvec2<valType> const & x
	)
    {
        return detail::tvec2<valType>(
            acos(x.x),
            acos(x.y));
    }

    template <typename valType>
	inline detail::tvec3<valType> acos
	(
		detail::tvec3<valType> const & x
	)
    {
        return detail::tvec3<valType>(
            acos(x.x),
            acos(x.y),
            acos(x.z));
    }

    template <typename valType>
	inline detail::tvec4<valType> acos
	(
		detail::tvec4<valType> const & x
	)
    {
        return detail::tvec4<valType>(
            acos(x.x),
            acos(x.y),
            acos(x.z),
            acos(x.w));
    }

    // atan
    template <typename genType>
    inline genType atan
	(
		genType const & y, 
		genType const & x
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        return ::std::atan2(y, x);
    }

    template <typename valType>
    inline detail::tvec2<valType> atan
	(
		detail::tvec2<valType> const & y, 
		detail::tvec2<valType> const & x
	)
    {
        return detail::tvec2<valType>(
            atan(y.x, x.x),
            atan(y.y, x.y));
    }

    template <typename valType>
    inline detail::tvec3<valType> atan
	(
		detail::tvec3<valType> const & y, 
		detail::tvec3<valType> const & x
	)
    {
        return detail::tvec3<valType>(
            atan(y.x, x.x),
            atan(y.y, x.y),
            atan(y.z, x.z));
    }

    template <typename valType>
    inline detail::tvec4<valType> atan
	(
		detail::tvec4<valType> const & y, 
		detail::tvec4<valType> const & x
	)
    {
        return detail::tvec4<valType>(
            atan(y.x, x.x),
            atan(y.y, x.y),
            atan(y.z, x.z),
            atan(y.w, x.w));
    }

    template <typename genType>
    inline genType atan
	(
		genType const & x
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        return ::std::atan(x);
    }

    template <typename valType>
    inline detail::tvec2<valType> atan
	(
		detail::tvec2<valType> const & x
	)
    {
        return detail::tvec2<valType>(
            atan(x.x),
            atan(x.y));
    }

    template <typename valType>
    inline detail::tvec3<valType> atan
	(
		detail::tvec3<valType> const & x
	)
    {
        return detail::tvec3<valType>(
            atan(x.x),
            atan(x.y),
            atan(x.z));
    }

    template <typename valType>
    inline detail::tvec4<valType> atan
	(
		detail::tvec4<valType> const & x
	)
    {
        return detail::tvec4<valType>(
            atan(x.x),
            atan(x.y),
            atan(x.z),
            atan(x.w));
    }

    // sinh
    template <typename genType> 
    inline genType sinh
	(
		genType const & angle
	)
    {
        GLM_STATIC_ASSERT(detail::type<genType>::is_float);

		return std::sinh(angle);
    }

    template <typename valType> 
    inline detail::tvec2<valType> sinh
	(
		detail::tvec2<valType> const & angle
	)
    {
        return detail::tvec2<valType>(
            sinh(angle.x),
            sinh(angle.y));
    }

    template <typename valType> 
    inline detail::tvec3<valType> sinh
	(
		detail::tvec3<valType> const & angle
	)
    {
        return detail::tvec3<valType>(
            sinh(angle.x),
            sinh(angle.y),
            sinh(angle.z));
    }

    template <typename valType> 
    inline detail::tvec4<valType> sinh
	(
		detail::tvec4<valType> const & angle
	)
    {
        return detail::tvec4<valType>(
            sinh(angle.x),
            sinh(angle.y),
            sinh(angle.z),
            sinh(angle.w));
    }

    // cosh
    template <typename genType> 
    inline genType cosh
	(
		genType const & angle
	)
    {
        GLM_STATIC_ASSERT(detail::type<genType>::is_float);

		return std::cosh(angle);
    }

    template <typename valType> 
    inline detail::tvec2<valType> cosh
	(
		detail::tvec2<valType> const & angle
	)
    {
        return detail::tvec2<valType>(
            cosh(angle.x),
            cosh(angle.y));
    }

    template <typename valType> 
    inline detail::tvec3<valType> cosh
	(
		detail::tvec3<valType> const & angle
	)
    {
        return detail::tvec3<valType>(
            cosh(angle.x),
            cosh(angle.y),
            cosh(angle.z));
    }

    template <typename valType> 
    inline detail::tvec4<valType> cosh
	(
		detail::tvec4<valType> const & angle
	)
    {
        return detail::tvec4<valType>(
            cosh(angle.x),
            cosh(angle.y),
            cosh(angle.z),
            cosh(angle.w));
    }

    // tanh
    template <typename genType>
    inline genType tanh
	(
		genType const & angle
	)
    {
        GLM_STATIC_ASSERT(detail::type<genType>::is_float);

		return std::tanh(angle);
    }

    template <typename valType> 
    inline detail::tvec2<valType> tanh
	(
		detail::tvec2<valType> const & angle
	)
    {
        return detail::tvec2<valType>(
            tanh(angle.x),
            tanh(angle.y));
    }

    template <typename valType> 
    inline detail::tvec3<valType> tanh
	(
		detail::tvec3<valType> const & angle
	)
    {
        return detail::tvec3<valType>(
            tanh(angle.x),
            tanh(angle.y),
            tanh(angle.z));
    }

    template <typename valType> 
    inline detail::tvec4<valType> tanh
	(
		detail::tvec4<valType> const & angle
	)
    {
        return detail::tvec4<valType>(
            tanh(angle.x),
            tanh(angle.y),
            tanh(angle.z),
            tanh(angle.w));
    }

    // asinh
    template <typename genType> 
    inline genType asinh
	(
		genType const & x
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);
		
		return (x < genType(0) ? genType(-1) : (x > genType(0) ? genType(1) : genType(0))) * log(abs(x) + sqrt(genType(1) + x * x));
    }

    template <typename valType> 
    inline detail::tvec2<valType> asinh
	(
		detail::tvec2<valType> const & x
	)
    {
        return detail::tvec2<valType>(
            asinh(x.x),
            asinh(x.y));
    }

    template <typename valType> 
    inline detail::tvec3<valType> asinh
	(
		detail::tvec3<valType> const & x
	)
    {
        return detail::tvec3<valType>(
            asinh(x.x),
            asinh(x.y),
            asinh(x.z));
    }

    template <typename valType> 
    inline detail::tvec4<valType> asinh
	(
		detail::tvec4<valType> const & x
	)
    {
        return detail::tvec4<valType>(
            asinh(x.x),
            asinh(x.y),
            asinh(x.z),
            asinh(x.w));
    }

    // acosh
    template <typename genType> 
    inline genType acosh
	(
		genType const & x
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

		if(x < genType(1))
			return genType(0);
		return log(x + sqrt(x * x - genType(1)));
    }

	template <typename valType> 
	inline detail::tvec2<valType> acosh
	(
		detail::tvec2<valType> const & x
	)
	{
		return detail::tvec2<valType>(
			acosh(x.x),
			acosh(x.y));
	}

    template <typename valType> 
    inline detail::tvec3<valType> acosh
	(
		detail::tvec3<valType> const & x
	)
    {
        return detail::tvec3<valType>(
            acosh(x.x),
            acosh(x.y),
            acosh(x.z));
    }

    template <typename valType> 
    inline detail::tvec4<valType> acosh
	(
		detail::tvec4<valType> const & x
	)
    {
        return detail::tvec4<valType>(
            acosh(x.x),
            acosh(x.y),
            acosh(x.z),
            acosh(x.w));
    }

    // atanh
    template <typename genType>
    inline genType atanh
	(
		genType const & x
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);
		
		if(abs(x) >= genType(1))
			return 0;
		return genType(0.5) * log((genType(1) + x) / (genType(1) - x));
    }

    template <typename valType> 
    inline detail::tvec2<valType> atanh
	(
		detail::tvec2<valType> const & x
	)
    {
        return detail::tvec2<valType>(
            atanh(x.x),
            atanh(x.y));
    }

    template <typename valType> 
    inline detail::tvec3<valType> atanh
	(
		detail::tvec3<valType> const & x
	)
    {
        return detail::tvec3<valType>(
            atanh(x.x),
            atanh(x.y),
            atanh(x.z));
    }

    template <typename valType> 
    inline detail::tvec4<valType> atanh
	(
		detail::tvec4<valType> const & x
	)
    {
        return detail::tvec4<valType>(
            atanh(x.x),
            atanh(x.y),
            atanh(x.z),
            atanh(x.w));
    }

	}//namespace trigonometric
	}//namespace function
	}//namespace core
}//namespace glm
