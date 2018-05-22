///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-05-21
// Updated : 2009-06-04
// Licence : This source is under MIT License
// File    : glm/gtc/quaternion.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <limits>

namespace glm{
namespace detail{

    template <typename valType> 
    inline tquat<valType>::tquat() : 
        x(0),
        y(0),
        z(0),
        w(1)
    {}

    template <typename valType> 
    inline tquat<valType>::tquat
	(
		valType const & s, 
		tvec3<valType> const & v
	) : 
        x(v.x),
        y(v.y),
        z(v.z),
        w(s)
    {}

    template <typename valType> 
    inline tquat<valType>::tquat
	(
		valType const & w, 
		valType const & x, 
		valType const & y, 
		valType const & z
	) :
        x(x),
        y(y),
        z(z),
        w(w)
    {}

    //////////////////////////////////////////////////////////////
    // tquat conversions

	//template <typename valType> 
	//inline tquat<valType>::tquat
	//(
	//	valType const & pitch, 
	//	valType const & yaw, 
	//	valType const & roll
	//)
	//{
	//	tvec3<valType> eulerAngle(pitch * valType(0.5), yaw * valType(0.5), roll * valType(0.5));
	//	tvec3<valType> c = glm::cos(eulerAngle * valType(0.5));
	//	tvec3<valType> s = glm::sin(eulerAngle * valType(0.5));
	//	
	//	this->w = c.x * c.y * c.z + s.x * s.y * s.z;
	//	this->x = s.x * c.y * c.z - c.x * s.y * s.z;
	//	this->y = c.x * s.y * c.z + s.x * c.y * s.z;
	//	this->z = c.x * c.y * s.z - s.x * s.y * c.z;
	//}

	template <typename valType> 
	inline tquat<valType>::tquat
	(
		tvec3<valType> const & eulerAngle
	)
	{
		tvec3<valType> c = glm::cos(eulerAngle * valType(0.5));
		tvec3<valType> s = glm::sin(eulerAngle * valType(0.5));
		
		this->w = c.x * c.y * c.z + s.x * s.y * s.z;
		this->x = s.x * c.y * c.z - c.x * s.y * s.z;
		this->y = c.x * s.y * c.z + s.x * c.y * s.z;
		this->z = c.x * c.y * s.z - s.x * s.y * c.z;		
	}

    template <typename valType> 
    inline tquat<valType>::tquat
	(
		tmat3x3<valType> const & m
	)
    {
		*this = toQuat(m);
    }

    template <typename valType> 
    inline tquat<valType>::tquat
	(
		tmat4x4<valType> const & m
	)
    {
		*this = toQuat(m);
    }

    //////////////////////////////////////////////////////////////
    // tquat<T> accesses

    template <typename valType> 
    inline valType& tquat<valType>::operator [] (int i)
    {
        return (&x)[i];
    }

    template <typename valType> 
    inline valType tquat<valType>::operator [] (int i) const
    {
        return (&x)[i];
    }

    //////////////////////////////////////////////////////////////
    // tquat<valType> operators

    template <typename valType> 
    inline tquat<valType>& tquat<valType>::operator *=
	(
		valType const & s
	)
    {
        this->w *= s;
        this->x *= s;
        this->y *= s;
        this->z *= s;
        return *this;
    }

    template <typename valType> 
    inline tquat<valType>& tquat<valType>::operator /=
	(
		valType const & s
	)
    {
        this->w /= s;
        this->x /= s;
        this->y /= s;
        this->z /= s;
        return *this;
    }

    //////////////////////////////////////////////////////////////
    // tquat<valType> external operators

	template <typename valType>
	inline detail::tquat<valType> operator- 
	(
		detail::tquat<valType> const & q
	)
	{
		return detail::tquat<valType>(-q.w, -q.x, -q.y, -q.z);
	}

	// Transformation
	template <typename valType>
	inline detail::tvec3<valType> operator* 
	(
		detail::tquat<valType> const & q, 
		detail::tvec3<valType> const & v
	)
	{
		detail::tvec3<valType> uv, uuv;
		detail::tvec3<valType> QuatVector(q.x, q.y, q.z);
		uv = glm::cross(QuatVector, v);
		uuv = glm::cross(QuatVector, uv);
		uv *= (valType(2) * q.w); 
		uuv *= valType(2); 

		return v + uv + uuv;
	}

	template <typename valType>
	inline detail::tvec3<valType> operator* 
	(
		detail::tvec3<valType> const & v,
		detail::tquat<valType> const & q 
	)
	{
		return gtc::quaternion::inverse(q) * v;
	}

	template <typename valType>
	inline detail::tvec4<valType> operator* 
	(
		detail::tquat<valType> const & q, 
		detail::tvec4<valType> const & v
	)
	{
		return detail::tvec4<valType>(q * detail::tvec3<valType>(v), v.w);
	}

	template <typename valType>
	inline detail::tvec4<valType> operator* 
	(
		detail::tvec4<valType> const & v,
		detail::tquat<valType> const & q 
	)
	{
		return gtc::quaternion::inverse(q) * v;
	}

	template <typename valType> 
	inline detail::tquat<valType> operator* 
	(
		detail::tquat<valType> const & q, 
		valType const & s
	)
	{
		return detail::tquat<valType>(
			q.w * s, q.x * s, q.y * s, q.z * s);
	}

	template <typename valType> 
	inline detail::tquat<valType> operator* 
	(
		valType const & s,
		detail::tquat<valType> const & q
	)
	{
		return q * s;
	}

	template <typename valType> 
	inline detail::tquat<valType> operator/ 
	(
		detail::tquat<valType> const & q, 
		valType const & s
	)
	{
		return detail::tquat<valType>(
			q.w / s, q.x / s, q.y / s, q.z / s);
	}

}//namespace detail

namespace gtc{
namespace quaternion{

	////////////////////////////////////////////////////////
    template <typename valType> 
	inline valType length
	(
		detail::tquat<valType> const & q
	)
    {
		return static_cast<valType>(glm::sqrt(dot(q, q)));
    }

    template <typename T> 
    inline detail::tquat<T> normalize
	(
		detail::tquat<T> const & q
	)
    {
        T len = static_cast<T>(length(q));
        if(len <= 0) // Problem
            return detail::tquat<T>(1, 0, 0, 0);
        T oneOverLen = 1 / len;
        return detail::tquat<T>(q.w * oneOverLen, q.x * oneOverLen, q.y * oneOverLen, q.z * oneOverLen);
    }

    template <typename valType> 
    inline valType dot
	(
		detail::tquat<valType> const & q1, 
		detail::tquat<valType> const & q2
	)
    {
        return q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;
    }

    template <typename valType> 
    inline detail::tquat<valType> cross
	(
		detail::tquat<valType> const & q1, 
		detail::tquat<valType> const & q2
	)
    {
        return detail::tquat<valType>(
            q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
	        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
	        q1.w * q2.y + q1.y * q2.w + q1.z * q2.x - q1.x * q2.z,
	        q1.w * q2.z + q1.z * q2.w + q1.x * q2.y - q1.y * q2.x);
    }

    template <typename T>
    inline detail::tquat<T> mix
	(
		detail::tquat<T> const & x, 
		detail::tquat<T> const & y, 
		T const & a
	)
    {
        if(a <= T(0)) return x;
        if(a >= T(1)) return y;

        float fCos = dot(x, y);
        detail::tquat<T> y2(y); //BUG!!! tquat<T> y2;
        if(fCos < T(0))
        {
            y2 = -y;
            fCos = -fCos;
        }

        //if(fCos > 1.0f) // problem
        float k0, k1;
        if(fCos > T(0.9999))
        {
            k0 = T(1) - a;
            k1 = T(0) + a; //BUG!!! 1.0f + a;
        }
        else
        {
            T fSin = sqrt(T(1) - fCos * fCos);
            T fAngle = atan(fSin, fCos);
            T fOneOverSin = T(1) / fSin;
            k0 = sin((T(1) - a) * fAngle) * fOneOverSin;
            k1 = sin((T(0) + a) * fAngle) * fOneOverSin;
        }

        return detail::tquat<T>(
            k0 * x.w + k1 * y2.w,
            k0 * x.x + k1 * y2.x,
            k0 * x.y + k1 * y2.y,
            k0 * x.z + k1 * y2.z);
    }

    template <typename valType> 
    inline detail::tquat<valType> conjugate
	(
		detail::tquat<valType> const & q
	)
    {
        return detail::tquat<valType>(q.w, -q.x, -q.y, -q.z);
    }

    template <typename valType> 
    inline detail::tquat<valType> inverse
	(
		detail::tquat<valType> const & q
	)
    {
        return gtc::quaternion::conjugate(q) / gtc::quaternion::length(q);
    }

    template <typename valType> 
    inline detail::tquat<valType> rotate
	(
		detail::tquat<valType> const & q, 
		valType const & angle, 
		detail::tvec3<valType> const & v
	)
    {
		detail::tvec3<valType> Tmp = v;

        // Axis of rotation must be normalised
        valType len = glm::core::function::geometric::length(Tmp);
        if(abs(len - valType(1)) > valType(0.001))
        {
            valType oneOverLen = valType(1) / len;
            Tmp.x *= oneOverLen;
            Tmp.y *= oneOverLen;
            Tmp.z *= oneOverLen;
        }

        valType AngleRad = radians(angle);
        valType fSin = sin(AngleRad * valType(0.5));

        return gtc::quaternion::cross(q, detail::tquat<valType>(cos(AngleRad * valType(0.5)), Tmp.x * fSin, Tmp.y * fSin, Tmp.z * fSin));
	}

    template <typename valType> 
    inline detail::tmat3x3<valType> mat3_cast
	(
		detail::tquat<valType> const & q
	)
    {
        detail::tmat3x3<valType> Result(valType(1));
        Result[0][0] = 1 - 2 * q.y * q.y - 2 * q.z * q.z;
        Result[0][1] = 2 * q.x * q.y + 2 * q.w * q.z;
        Result[0][2] = 2 * q.x * q.z - 2 * q.w * q.y;

        Result[1][0] = 2 * q.x * q.y - 2 * q.w * q.z;
        Result[1][1] = 1 - 2 * q.x * q.x - 2 * q.z * q.z;
        Result[1][2] = 2 * q.y * q.z + 2 * q.w * q.x;

        Result[2][0] = 2 * q.x * q.z + 2 * q.w * q.y;
        Result[2][1] = 2 * q.y * q.z - 2 * q.w * q.x;
        Result[2][2] = 1 - 2 * q.x * q.x - 2 * q.y * q.y;
        return Result;
    }

    template <typename valType> 
    inline detail::tmat4x4<valType> mat4_cast
	(
		detail::tquat<valType> const & q
	)
    {
        return detail::tmat4x4<valType>(mat3_cast(q));
    }

    template <typename T> 
    inline detail::tquat<T> quat_cast
	(
		detail::tmat3x3<T> const & m
	)
    {
        T fourXSquaredMinus1 = m[0][0] - m[1][1] - m[2][2];
        T fourYSquaredMinus1 = m[1][1] - m[0][0] - m[2][2];
        T fourZSquaredMinus1 = m[2][2] - m[0][0] - m[1][1];
        T fourWSquaredMinus1 = m[0][0] + m[1][1] + m[2][2];
        
        int biggestIndex = 0;
        T fourBiggestSquaredMinus1 = fourWSquaredMinus1;
        if(fourXSquaredMinus1 > fourBiggestSquaredMinus1)
        {
            fourBiggestSquaredMinus1 = fourXSquaredMinus1;
            biggestIndex = 1;
        }
        if(fourYSquaredMinus1 > fourBiggestSquaredMinus1)
        {
            fourBiggestSquaredMinus1 = fourYSquaredMinus1;
            biggestIndex = 2;
        }
        if(fourZSquaredMinus1 > fourBiggestSquaredMinus1)
        {
            fourBiggestSquaredMinus1 = fourZSquaredMinus1;
            biggestIndex = 3;
        }

        T biggestVal = sqrt(fourBiggestSquaredMinus1 + T(1)) * T(0.5);
        T mult = T(0.25) / biggestVal;

        detail::tquat<T> Result;
        switch(biggestIndex)
        {
        case 0:
            Result.w = biggestVal; 
            Result.x = (m[1][2] - m[2][1]) * mult;
            Result.y = (m[2][0] - m[0][2]) * mult;
            Result.z = (m[0][1] - m[1][0]) * mult;
            break;
        case 1:
            Result.w = (m[1][2] - m[2][1]) * mult;
            Result.x = biggestVal;
            Result.y = (m[0][1] + m[1][0]) * mult;
            Result.z = (m[2][1] + m[1][2]) * mult;
            break;
        case 2:
            Result.w = (m[2][0] - m[0][2]) * mult;
            Result.x = (m[0][1] + m[1][0]) * mult;
            Result.y = biggestVal;
            Result.z = (m[1][2] + m[2][1]) * mult;
            break;
        case 3:
            Result.w = (m[0][1] - m[1][0]) * mult;
            Result.x = (m[2][0] + m[0][2]) * mult;
            Result.y = (m[1][2] + m[2][1]) * mult;
            Result.z = biggestVal;
            break;
        }
        return Result;
    }

    template <typename valType> 
    inline detail::tquat<valType> quat_cast
	(
		detail::tmat4x4<valType> const & m4
	)
    {
		return quat_cast(detail::tmat3x3<valType>(m4));
    }

}//namespace quaternion
}//namespace gtc
}//namespace glm
