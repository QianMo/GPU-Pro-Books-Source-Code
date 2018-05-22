///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-08-01
// Updated : 2008-09-23
// Licence : This source is under MIT License
// File    : glm/core/func_noise.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	namespace core{
	namespace function{
	namespace noise{

	// noise1
	template <typename genType>
	inline genType noise1
	(
		genType const & x
	)
	{
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

		int iNbr = int(x + genType(3) / genType(2)) * 1103515245 + 12345;
		return genType(int(iNbr / genType(65536)) % 32768) / genType(32767);
	}

	template <typename valType>
	inline typename detail::tvec2<valType>::value_type noise1
	(
		detail::tvec2<valType> const & x
	)
	{
		valType tmp(0);
		for(typename detail::tvec2<valType>::size_type i = 0; i < detail::tvec2<valType>::value_size(); ++i)
			tmp += x[i];
		return noise1(tmp);
	}

	template <typename valType>
	inline typename detail::tvec3<valType>::value_type noise1
	(
		detail::tvec3<valType> const & x
	)
	{
		valType tmp(0);
		for(typename detail::tvec3<valType>::size_type i = 0; i < detail::tvec3<valType>::value_size(); ++i)
			tmp += x[i];
		return noise1(tmp);
	}

	template <typename valType>
	inline typename detail::tvec4<valType>::value_type noise1
	(
		detail::tvec4<valType> const & x
	)
	{
		valType tmp(0);
		for(typename detail::tvec4<valType>::size_type i = 0; i < detail::tvec4<valType>::value_size(); ++i)
			tmp += x[i];
		return noise1(tmp);
	}

	// noise2
	template <typename genType>
	inline detail::tvec2<genType> noise2
	(
		genType const & x
	)
	{
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

		genType f1 = x * genType(1103515245) + genType(12345);
		genType f2 = f1 * genType(1103515245) + genType(12345);
		return detail::tvec2<genType>(
			noise1(f1),
			noise1(f2));
	}

	template <typename valType>
	inline detail::tvec2<valType> noise2
	(
		detail::tvec2<valType> const & x
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		valType f0(0);
		for(typename detail::tvec2<valType>::size_type i = 0; i < detail::tvec2<valType>::value_size(); ++i)
			f0 += x[i];
		
		valType f1 = f0 * valType(1103515245) + valType(12345);
		valType f2 = f1 * valType(1103515245) + valType(12345);
		return detail::tvec2<valType>(
			noise1(f1),
			noise1(f2));
	}

	template <typename valType>
	inline detail::tvec2<valType> noise2
	(
		detail::tvec3<valType> const & x
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		valType f0(0);
		for(typename detail::tvec3<valType>::size_type i = 0; i < detail::tvec3<valType>::value_size(); ++i)
			f0 += x[i];

		valType f1 = f0 * valType(1103515245) + valType(12345);
		valType f2 = f1 * valType(1103515245) + valType(12345);
		return detail::tvec2<valType>(
			noise1(f1),
			noise1(f2));
	}

	template <typename valType>
	inline detail::tvec2<valType> noise2
	(
		detail::tvec4<valType> const & x
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		valType f0(0);
		for(typename detail::tvec4<valType>::size_type i = 0; i < detail::tvec4<valType>::value_size(); ++i)
			f0 += x[i];
		
		valType f1 = f0 * valType(1103515245) + valType(12345);
		valType f2 = f1 * valType(1103515245) + valType(12345);
		return detail::tvec2<valType>(
			noise1(f1),
			noise1(f2));
	}

	// noise3
	template <typename genType>
	inline detail::tvec3<genType> noise3
	(
		genType const & x
	)
	{
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

		genType f1 = x * genType(1103515245) + genType(12345);
		genType f2 = f1 * genType(1103515245) + genType(12345);
		genType f3 = f2 * genType(1103515245) + genType(12345);
		return detail::tvec3<genType>(
			noise1(f1),
			noise1(f2),
			noise1(f3));
	}

	template <typename valType>
	inline detail::tvec3<valType> noise3
	(
		detail::tvec2<valType> const & x
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		valType f0(0);
		for(typename detail::tvec2<valType>::size_type i = 0; i < detail::tvec2<valType>::value_size(); ++i)
			f0 += x[i];
		valType f1 = f0 * valType(1103515245) + valType(12345);
		valType f2 = f1 * valType(1103515245) + valType(12345);
		valType f3 = f2 * valType(1103515245) + valType(12345);
		return detail::tvec3<valType>(
			noise1(f1),
			noise1(f2),
			noise1(f3));
	}

	template <typename valType>
	inline detail::tvec3<valType> noise3
	(
		detail::tvec3<valType> const & x
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		valType f0(0);
		for(typename detail::tvec3<valType>::size_type i = 0; i < detail::tvec3<valType>::value_size(); ++i)
			f0 += x[i];
		valType f1 = f0 * valType(1103515245) + valType(12345);
		valType f2 = f1 * valType(1103515245) + valType(12345);
		valType f3 = f2 * valType(1103515245) + valType(12345);
		return detail::tvec3<valType>(
			noise1(f1),
			noise1(f2),
			noise1(f3));
	}

	template <typename valType>
	inline detail::tvec3<valType> noise3
	(
		detail::tvec4<valType> const & x
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		valType f0(0);
		for(typename detail::tvec4<valType>::size_type i = 0; i < detail::tvec4<valType>::value_size(); ++i)
			f0 += x[i];
		valType f1 = f0 * valType(1103515245) + valType(12345);
		valType f2 = f1 * valType(1103515245) + valType(12345);
		valType f3 = f2 * valType(1103515245) + valType(12345);
		return detail::tvec3<valType>(
			noise1(f1),
			noise1(f2),
			noise1(f3));
	}

	// noise4
	template <typename genType>
	inline detail::tvec4<genType> noise4
	(
		genType const & x
	)
	{
		GLM_STATIC_ASSERT(detail::type<genType>::is_float);

		genType f1 = x * genType(1103515245) + genType(12345);
		genType f2 = f1 * genType(1103515245) + genType(12345);
		genType f3 = f2 * genType(1103515245) + genType(12345);
		genType f4 = f3 * genType(1103515245) + genType(12345);
		return detail::tvec4<genType>(
			noise1(f1),
			noise1(f2),
			noise1(f3),
			noise1(f4));
	}

	template <typename valType>
	inline detail::tvec4<valType> noise4
	(
		detail::tvec2<valType> const & x
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		valType f0(0);
		for(typename detail::tvec2<valType>::size_type i = 0; i < detail::tvec2<valType>::value_size(); ++i)
			f0 += x[i];
		valType f1 = f0 * valType(1103515245) + valType(12345);
		valType f2 = f1 * valType(1103515245) + valType(12345);
		valType f3 = f2 * valType(1103515245) + valType(12345);
		valType f4 = f3 * valType(1103515245) + valType(12345);
		return detail::tvec4<valType>(
			noise1(f1),
			noise1(f2),
			noise1(f3),
			noise1(f4));
	}

	template <typename valType>
	inline detail::tvec4<valType> noise4
	(
		detail::tvec3<valType> const & x
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		valType f0(0);
		for(typename detail::tvec3<valType>::size_type i = 0; i < detail::tvec3<valType>::value_size(); ++i)
			f0 += x[i];
		valType f1 = f0 * valType(1103515245) + valType(12345);
		valType f2 = f1 * valType(1103515245) + valType(12345);
		valType f3 = f2 * valType(1103515245) + valType(12345);
		valType f4 = f3 * valType(1103515245) + valType(12345);
		return detail::tvec4<valType>(
			noise1(f1),
			noise1(f2),
			noise1(f3),
			noise1(f4));
	}

	template <typename valType>
	inline detail::tvec4<valType> noise4
	(
		detail::tvec4<valType> const & x
	)
	{
		GLM_STATIC_ASSERT(detail::type<valType>::is_float);

		valType f0(0);
		for(typename detail::tvec4<valType>::size_type i = 0; i < detail::tvec4<valType>::value_size(); ++i)
			f0 += x[i];
		valType f1 = f0 * valType(1103515245) + valType(12345);
		valType f2 = f1 * valType(1103515245) + valType(12345);
		valType f3 = f2 * valType(1103515245) + valType(12345);
		valType f4 = f3 * valType(1103515245) + valType(12345);
		return detail::tvec4<valType>(
			noise1(f1),
			noise1(f2),
			noise1(f3),
			noise1(f4));
	}

	}//namespace noise
	}//namespace function
	}//namespace core
}//namespace glm
