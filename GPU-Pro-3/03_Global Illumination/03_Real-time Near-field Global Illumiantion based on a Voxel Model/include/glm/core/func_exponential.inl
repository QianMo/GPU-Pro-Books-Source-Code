///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-08-03
// Updated : 2008-09-06
// Licence : This source is under MIT License
// File    : glm/core/func_exponential.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	namespace core{
	namespace function{
	namespace exponential{

    // pow
    template <typename genType>
    inline genType pow
	(
		genType const & x, 
		genType const & y
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        return ::std::pow(x, y);
    }

    template <typename valType>
    inline detail::tvec2<valType> pow
	(
		detail::tvec2<valType> const & x, 
		detail::tvec2<valType> const & y
	)
    {
        return detail::tvec2<valType>(
            pow(x.x, y.x),
            pow(x.y, y.y));
    }

    template <typename valType>
    inline detail::tvec3<valType> pow
	(
		detail::tvec3<valType> const & x, 
		detail::tvec3<valType> const & y
	)
    {
        return detail::tvec3<valType>(
            pow(x.x, y.x),
            pow(x.y, y.y),
            pow(x.z, y.z));
    }

    template <typename valType>
    inline detail::tvec4<valType> pow
	(
		detail::tvec4<valType> const & x, 
		detail::tvec4<valType> const & y
	)
    {
        return detail::tvec4<valType>(
            pow(x.x, y.x),
            pow(x.y, y.y),
            pow(x.z, y.z),
            pow(x.w, y.w));
    }

    // exp
    template <typename genType>
    inline genType exp
	(
		genType const & x
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        return ::std::exp(x);
    }

    template <typename valType>
    inline detail::tvec2<valType> exp
	(
		detail::tvec2<valType> const & x
	)
    {
        return detail::tvec2<valType>(
            exp(x.x),
            exp(x.y));
    }

    template <typename valType>
    inline detail::tvec3<valType> exp
	(
		detail::tvec3<valType> const & x
	)
    {
        return detail::tvec3<valType>(
            exp(x.x),
            exp(x.y),
            exp(x.z));
    }

    template <typename valType>
    inline detail::tvec4<valType> exp
	(
		detail::tvec4<valType> const & x
	)
    {
        return detail::tvec4<valType>(
            exp(x.x),
            exp(x.y),
            exp(x.z),
            exp(x.w));
    }

    // log
    template <typename genType>
    inline genType log
	(
		genType const & x
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        return ::std::log(x);
    }

    template <typename valType>
    inline detail::tvec2<valType> log
	(
		detail::tvec2<valType> const & x
	)
    {
        return detail::tvec2<valType>(
            log(x.x),
            log(x.y));
    }

    template <typename valType>
    inline detail::tvec3<valType> log
	(
		detail::tvec3<valType> const & x
	)
    {
        return detail::tvec3<valType>(
            log(x.x),
            log(x.y),
            log(x.z));
    }

    template <typename valType>
    inline detail::tvec4<valType> log
	(
		detail::tvec4<valType> const & x
	)
    {
        return detail::tvec4<valType>(
            log(x.x),
            log(x.y),
            log(x.z),
            log(x.w));
    }

    //exp2, ln2 = 0.69314718055994530941723212145818f
    template <typename genType>
    inline genType exp2
	(
		genType const & x
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        return ::std::exp(genType(0.69314718055994530941723212145818) * x);
    }

    template <typename valType>
    inline detail::tvec2<valType> exp2
	(
		detail::tvec2<valType> const & x
	)
    {
        return detail::tvec2<valType>(
            exp2(x.x),
            exp2(x.y));
    }

    template <typename valType>
    inline detail::tvec3<valType> exp2
	(
		detail::tvec3<valType> const & x
	)
    {
        return detail::tvec3<valType>(
            exp2(x.x),
            exp2(x.y),
            exp2(x.z));
    }

    template <typename valType>
    inline detail::tvec4<valType> exp2
	(
		detail::tvec4<valType> const & x
	)
    {
        return detail::tvec4<valType>(
            exp2(x.x),
            exp2(x.y),
            exp2(x.z),
            exp2(x.w));
    }

    // log2, ln2 = 0.69314718055994530941723212145818f
    template <typename genType>
    inline genType log2
	(
		genType const & x
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        return ::std::log(x) / genType(0.69314718055994530941723212145818);
    }

    template <typename valType>
    inline detail::tvec2<valType> log2
	(
		const detail::tvec2<valType>& x
	)
    {
        return detail::tvec2<valType>(
            log2(x.x),
            log2(x.y));
    }

    template <typename valType>
    inline detail::tvec3<valType> log2
	(
		const detail::tvec3<valType>& x
	)
    {
        return detail::tvec3<valType>(
            log2(x.x),
            log2(x.y),
            log2(x.z));
    }

    template <typename valType>
    inline detail::tvec4<valType> log2
	(
		detail::tvec4<valType> const & x
	)
    {
        return detail::tvec4<valType>(
            log2(x.x),
            log2(x.y),
            log2(x.z),
            log2(x.w));
    }

    // sqrt
    template <typename genType>
    inline genType sqrt
	(
		genType const & x
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        return genType(::std::sqrt(double(x)));
    }

    template <typename valType>
    inline detail::tvec2<valType> sqrt
	(
		detail::tvec2<valType> const & x
	)
    {
        return detail::tvec2<valType>(
            sqrt(x.x),
            sqrt(x.y));
    }

    template <typename valType>
    inline detail::tvec3<valType> sqrt
	(
		detail::tvec3<valType> const & x
	)
    {
        return detail::tvec3<valType>(
            sqrt(x.x),
            sqrt(x.y),
            sqrt(x.z));
    }

    template <typename valType>
    inline detail::tvec4<valType> sqrt
	(
		detail::tvec4<valType> const & x
	)
    {
        return detail::tvec4<valType>(
            sqrt(x.x),
            sqrt(x.y),
            sqrt(x.z),
            sqrt(x.w));
    }

    template <typename genType>
    inline genType inversesqrt
	(
		genType const & x
	)
    {
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

        return genType(1) / ::std::sqrt(x);
    }

    template <typename valType>
    inline detail::tvec2<valType> inversesqrt
	(
		detail::tvec2<valType> const & x
	)
    {
        return detail::tvec2<valType>(
            inversesqrt(x.x),
            inversesqrt(x.y));
    }

    template <typename valType>
    inline detail::tvec3<valType> inversesqrt
	(
		detail::tvec3<valType> const & x
	)
    {
        return detail::tvec3<valType>(
            inversesqrt(x.x),
            inversesqrt(x.y),
            inversesqrt(x.z));
    }

    template <typename valType>
    inline detail::tvec4<valType> inversesqrt
	(
		detail::tvec4<valType> const & x
	)
    {
        return detail::tvec4<valType>(
            inversesqrt(x.x),
            inversesqrt(x.y),
            inversesqrt(x.z),
            inversesqrt(x.w));
    }

	}//namespace exponential
	}//namespace function
	}//namespace core
}//namespace glm
