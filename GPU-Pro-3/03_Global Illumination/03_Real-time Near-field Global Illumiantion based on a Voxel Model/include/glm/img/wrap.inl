///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-11-25
// Updated : 2009-11-25
// Licence : This source is under MIT License
// File    : glm/img/wrap.inl
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace img{
namespace wrap
{
	////////////////////////
	// clamp

	template <typename valType> 
	inline valType clamp
	(
		valType const & Texcoord
	)
	{
		return glm::clamp(Texcoord, valType(0), valType(1));
	}

	template <typename valType> 
	inline detail::tvec2<valType> clamp
	(
		detail::tvec2<valType> const & Texcoord
	)
	{
		detail::tvec2<valType> Result;
		for(typename detail::tvec2<valType>::size_type i = 0; i < detail::tvec2<valType>::value_size(); ++i)
			Result[i] = clamp(Texcoord[i]);
		return Result;
	}

	template <typename valType> 
	inline detail::tvec3<valType> clamp
	(
		detail::tvec3<valType> const & Texcoord
	)
	{
		detail::tvec3<valType> Result;
		for(typename detail::tvec3<valType>::size_type i = 0; i < detail::tvec3<valType>::value_size(); ++i)
			Result[i] = clamp(Texcoord[i]);
		return Result;
	}

	template <typename valType> 
	inline detail::tvec4<valType> clamp
	(
		detail::tvec4<valType> const & Texcoord
	)
	{
		detail::tvec4<valType> Result;
		for(typename detail::tvec4<valType>::size_type i = 0; i < detail::tvec4<valType>::value_size(); ++i)
			Result[i] = clamp(Texcoord[i]);
		return Result;
	}

	////////////////////////
	// repeat

	template <typename valType> 
	inline valType repeat
	(
		valType const & Texcoord
	)
	{
		return glm::fract(Texcoord);
	}

	template <typename valType> 
	inline detail::tvec2<valType> repeat
	(
		detail::tvec2<valType> const & Texcoord
	)
	{
		detail::tvec2<valType> Result;
		for(typename detail::tvec2<valType>::size_type i = 0; i < detail::tvec2<valType>::value_size(); ++i)
			Result[i] = repeat(Texcoord[i]);
		return Result;
	}

	template <typename valType> 
	inline detail::tvec3<valType> repeat
	(
		detail::tvec3<valType> const & Texcoord
	)
	{
		detail::tvec3<valType> Result;
		for(typename detail::tvec3<valType>::size_type i = 0; i < detail::tvec3<valType>::value_size(); ++i)
			Result[i] = repeat(Texcoord[i]);
		return Result;
	}

	template <typename valType> 
	inline detail::tvec4<valType> repeat
	(
		detail::tvec4<valType> const & Texcoord
	)
	{
		detail::tvec4<valType> Result;
		for(typename detail::tvec4<valType>::size_type i = 0; i < detail::tvec4<valType>::value_size(); ++i)
			Result[i] = repeat(Texcoord[i]);
		return Result;
	}

	////////////////////////
	// mirrorRepeat

	template <typename genType> 
	inline genType mirrorRepeat
	(
		genType const & Texcoord
	)
	{
		genType const Clamp = genType(int(glm::floor(Texcoord)) % 2);
		genType const Floor = glm::floor(Texcoord);
		genType const Rest = Texcoord - Floor;
		genType const Mirror = Clamp + Rest;

		genType Out;
		if(Mirror >= genType(1))
			Out = genType(1) - Rest;
		else
			Out = Rest;
		return Out;
	}

	template <typename valType> 
	inline detail::tvec2<valType> mirrorRepeat
	(
		detail::tvec2<valType> const & Texcoord
	)
	{
		detail::tvec2<valType> Result;
		for(typename detail::tvec2<valType>::size_type i = 0; i < detail::tvec2<valType>::value_size(); ++i)
			Result[i] = mirrorRepeat(Texcoord[i]);
		return Result;
	}

	template <typename valType> 
	inline detail::tvec3<valType> mirrorRepeat
	(
		detail::tvec3<valType> const & Texcoord
	)
	{
		detail::tvec3<valType> Result;
		for(typename detail::tvec3<valType>::size_type i = 0; i < detail::tvec3<valType>::value_size(); ++i)
			Result[i] = mirrorRepeat(Texcoord[i]);
		return Result;
	}

	template <typename valType> 
	inline detail::tvec4<valType> mirrorRepeat
	(
		detail::tvec4<valType> const & Texcoord
	)
	{
		detail::tvec4<valType> Result;
		for(typename detail::tvec4<valType>::size_type i = 0; i < detail::tvec4<valType>::value_size(); ++i)
			Result[i] = mirrorRepeat(Texcoord[i]);
		return Result;
	}

}//namespace wrap
}//namespace img
}//namespace glm
