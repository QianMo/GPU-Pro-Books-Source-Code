///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2007-03-14
// Updated : 2007-03-14
// Licence : This source is under MIT License
// File    : gtx_extented_min_max.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace gtx{
namespace extented_min_max
{
	template <typename T> 
	inline T min(
		const T x, 
		const T y, 
		const T z)
	{
		return min(min(x, y), z);
	}

	template <typename T> 
	inline T min(
		const T x, 
		const T y, 
		const T z, 
		const T w)
	{
		return min(min(x, y), min(z, w));
	}

	template <typename T> 
	inline detail::tvec2<T> min(
		const detail::tvec2<T>& x, 
		const T y, 
		const T z)
	{
		return min(x, min(y, z));
	}

    template <typename T> 
	inline detail::tvec3<T> min(
		const detail::tvec3<T>& x, 
		const T y, 
		const T z)
	{
		return min(x, min(y, z));
	}

    template <typename T> 
	inline detail::tvec4<T> min(
		const detail::tvec4<T>& x, 
		const T y, 
		const T z)
	{
		return min(x, min(y, z));
	}

    template <typename T> 
	inline detail::tvec2<T> min(
		const detail::tvec2<T>& x, 
		const T y, 
		const T z, 
		const T w)
	{
		return min(x, min(y, min(z, w)));
	}

    template <typename T> 
	inline detail::tvec3<T> min(
		const detail::tvec3<T>& x, 
		const T y, 
		const T z, 
		const T w)
	{
		return min(x, min(y, min(z, w)));
	}

    template <typename T> 
	inline detail::tvec4<T> min(
		const detail::tvec4<T>& x, 
		const T y, 
		const T z, 
		const T w)
	{
		return min(x, min(y, min(z, w)));
	}

    template <typename T> 
	inline detail::tvec2<T> min(
		const detail::tvec2<T>& x, 
		const detail::tvec2<T>& y, 
		const detail::tvec2<T>& z)
	{
		return min(x, min(y, z));
	}

    template <typename T> 
	inline detail::tvec3<T> min(
		const detail::tvec3<T>& x, 
		const detail::tvec3<T>& y, 
		const detail::tvec3<T>& z)
	{
		return min(x, min(y, z));
	}

    template <typename T> 
	inline detail::tvec4<T> min(
		const detail::tvec4<T>& x, 
		const detail::tvec4<T>& y, 
		const detail::tvec4<T>& z)
	{
		return min(x, min(y, z));
	}
	
    template <typename T> 
	inline detail::tvec2<T> min(
		const detail::tvec2<T>& x, 
		const detail::tvec2<T>& y, 
		const detail::tvec2<T>& z, 
		const detail::tvec2<T>& w)
	{
		return min(min(x, y), min(z, w));
	}

    template <typename T> 
	inline detail::tvec3<T> min(
		const detail::tvec3<T>& x, 
		const detail::tvec3<T>& y, 
		const detail::tvec3<T>& z, 
		const detail::tvec3<T>& w)
	{
		return min(min(x, y), min(z, w));
	}

    template <typename T> 
	inline detail::tvec4<T> min(
		const detail::tvec4<T>& x, 
		const detail::tvec4<T>& y, 
		const detail::tvec4<T>& z, 
		const detail::tvec4<T>& w)
	{
		return min(min(x, y), min(z, w));
	}

	template <typename T> 
	inline T max(
		const T x, 
		const T y, 
		const T z)
	{
		return max(max(x, y), z);
	}

	template <typename T> 
	inline T max(
		const T x, 
		const T y, 
		const T z, 
		const T w)
	{
		return max(max(x, y), max(z, w));
	}

	template <typename T> 
	inline detail::tvec2<T> max(
		const detail::tvec2<T>& x, 
		const T y, 
		const T z)
	{
		return max(x, max(y, z));
	}

    template <typename T> 
	inline detail::tvec3<T> max(
		const detail::tvec3<T>& x, 
		const T y, 
		const T z)
	{
		return max(x, max(y, z));
	}

    template <typename T> 
	inline detail::tvec4<T> max(
		const detail::tvec4<T>& x, 
		const T y, 
		const T z)
	{
		return max(x, max(y, z));
	}

    template <typename T> 
	inline detail::tvec2<T> max(
		const detail::tvec2<T>& x, 
		const T y, 
		const T z, 
		const T w)
	{
		return max(max(x, y), max(z, w));
	}

    template <typename T> 
	inline detail::tvec3<T> max(
		const detail::tvec3<T>& x, 
		const T y, 
		const T z, 
		const T w)
	{
		return max(max(x, y), max(z, w));
	}

    template <typename T> 
	inline detail::tvec4<T> max(
		const detail::tvec4<T>& x, 
		const T y, 
		const T z, 
		const T w)
	{
		return max(max(x, y), max(z, w));
	}
	
    template <typename T> 
	inline detail::tvec2<T> max(
		const detail::tvec2<T>& x, 
		const detail::tvec2<T>& y, 
		const detail::tvec2<T>& z)
	{
		return max(max(x, y), z);
	}

    template <typename T> 
	inline detail::tvec3<T> max(
		const detail::tvec3<T>& x, 
		const detail::tvec3<T>& y, 
		const detail::tvec3<T>& z)
	{
		return max(max(x, y), z);
	}

    template <typename T> 
	inline detail::tvec4<T> max(
		const detail::tvec4<T>& x, 
		const detail::tvec4<T>& y, 
		const detail::tvec4<T>& z)
	{
		return max(max(x, y), z);
	}

    template <typename T> 
	inline detail::tvec2<T> max(
		const detail::tvec2<T>& x, 
		const detail::tvec2<T>& y, 
		const detail::tvec2<T>& z, 
		const detail::tvec2<T>& w)
	{
		return max(max(x, y), max(z, w));
	}

    template <typename T> 
	inline detail::tvec3<T> max(
		const detail::tvec3<T>& x, 
		const detail::tvec3<T>& y, 
		const detail::tvec3<T>& z, 
		const detail::tvec3<T>& w)
	{
		return max(max(x, y), max(z, w));
	}

    template <typename T> 
	inline detail::tvec4<T> max(
		const detail::tvec4<T>& x, 
		const detail::tvec4<T>& y, 
		const detail::tvec4<T>& z, 
		const detail::tvec4<T>& w)
	{
		return max(max(x, y), max(z, w));
	}

}//namespace extented_min_max
}//namespace gtx
}//namespace glm
