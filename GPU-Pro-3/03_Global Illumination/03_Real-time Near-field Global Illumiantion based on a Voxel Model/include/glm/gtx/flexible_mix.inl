///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2007-09-21
// Updated : 2007-09-21
// Licence : This source is under MIT licence
// File    : glm/gtx/flexible_mix.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
    // mix
    template <typename T, typename U>
    inline T mixGTX(T x, T y, U a)
    {
		//GLM_STATIC_ASSERT(detail::traits<U>::is_float);
		//return T(x * (U(1) - a) + y * a);
		return T(x + a * (y - x));
    }

	template <typename T, typename U>
    inline detail::tvec2<T> mixGTX(const detail::tvec2<T>& x, const detail::tvec2<T>& y, U a)
    {
		return detail::tvec2<T>(detail::tvec2<U>(x) * (U(1) - a) + detail::tvec2<U>(y) * a);
        //return x * (U(1) - a) + y * a;
    }

    template <typename T, typename U>
    inline detail::tvec3<T> mixGTX(const detail::tvec3<T>& x, const detail::tvec3<T>& y, U a)
    {
		return detail::tvec3<T>(detail::tvec3<U>(x) * (U(1) - a) + detail::tvec3<U>(y) * a);
        //return x * (U(1) - a) + y * a;
		//return mix(x, y, tvec3<U>(a));
    }

    template <typename T, typename U>
    inline detail::tvec4<T> mixGTX(const detail::tvec4<T>& x, const detail::tvec4<T>& y, U a)
    {
		return detail::tvec4<T>(detail::tvec4<U>(x) * (U(1) - a) + detail::tvec4<U>(y) * a);
        //return x * (U(1) - a) + y * a;
    }

    template <typename T, typename U>
    inline detail::tvec2<T> mixGTX(const detail::tvec2<T>& x, const detail::tvec2<T>& y, const detail::tvec2<U>& a)
    {
		return detail::tvec2<T>(detail::tvec2<U>(x) * (U(1) - a) + detail::tvec2<U>(y) * a);
    }

    template <typename T, typename U>
    inline detail::tvec3<T> mixGTX(const detail::tvec3<T>& x, const detail::tvec3<T>& y, const detail::tvec3<U>& a)
    {
        return detail::tvec3<T>(detail::tvec3<U>(x) * (U(1) - a) + detail::tvec3<U>(y) * a);
    }

    template <typename T, typename U>
    inline detail::tvec4<T> mixGTX(const detail::tvec4<T>& x, const detail::tvec4<T>& y, const detail::tvec4<U>& a)
    {
		return detail::tvec4<T>(detail::tvec4<U>(x) * (U(1) - a) + detail::tvec4<U>(y) * a);
    }
}
