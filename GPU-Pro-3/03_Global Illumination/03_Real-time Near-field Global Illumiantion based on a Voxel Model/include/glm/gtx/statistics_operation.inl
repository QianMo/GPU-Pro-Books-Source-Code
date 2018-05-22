///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2007-11-21
// Updated : 2007-11-21
// Licence : This source is under MIT License
// File    : glm/gtx/statistics_operator.inl
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cassert>

namespace glm
{
    //! Compute the sum of square of differences between each matrices paremeters
    template <typename T>
    inline T statDistanceGTX(const detail::tmat2x2<T>& m1, const detail::tmat2x2<T>& m2)
    {
        T result = T(0);
        for(int j = 0; j < 2; ++j)
        for(int i = 0; i < 2; ++i)
        {
            T diff = m1[j][i] - m2[j][i];
            result += diff * diff;
        }
        return result;
    }

    template <typename T>
    inline T statDistanceGTX(const detail::tmat3x3<T>& m1, const detail::tmat3x3<T>& m2)
    {
        T result = T(0);
        for(int j = 0; j < 3; ++j)
        for(int i = 0; i < 3; ++i)
        {
            T diff = m1[j][i] - m2[j][i];
            result += diff * diff;
        }
        return result;
    }

    template <typename T>
    inline T statDistanceGTX(const detail::tmat4x4<T>& m1, const detail::tmat4x4<T>& m2)
    {
        T result = T(0);
        for(int j = 0; j < 4; ++j)
        for(int i = 0; i < 4; ++i)
        {
            T diff = m1[j][i] - m2[j][i];
            result += diff * diff;
        }
        return result;
    }

    template <typename T> 
    T expectedValueGTX(const detail::tmat4x4<T>& m)
    {
        T result = T(0);
        for(int j = 0; j < 4; ++j)
        for(int i = 0; i < 4; ++i)
            result += m[j][i];
        result *= T(0,0625);
        return result;
    }

    template <typename T> 
    T varianceGTX(const detail::tmat4x4<T>& m)
    {
        T ExpectedValue = expectedValueGTX(m);
        T ExpectedValueOfSquaredMatrix = expectedValueGTX(matrixCompMult(m));
        return ExpectedValueOfSquaredMatrix - ExpectedValue * ExpectedValue;
    }

    template <typename T> 
    T standardDevitionGTX(const detail::tmat4x4<T>& m)
    {
        return sqrt(varianceGTX(m));
    }
}
