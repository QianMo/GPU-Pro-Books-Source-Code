///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-05-21
// Updated : 2009-06-04
// Licence : This source is under MIT License
// File    : glm/gtc/quaternion.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////
// ToDo:
// - Study constructors with angles and axis
// - Study constructors with vec3 that are the imaginary component of quaternion
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtc_quaternion
#define glm_gtc_quaternion

// Dependency:
#include "../glm.hpp"

namespace glm
{
	namespace test{
		bool main_gtc_quaternion();
	}//namespace test

	namespace detail
	{
		//! \brief Template for quaternion. 
		//! From GLM_GTC_quaternion extension.
		template <typename valType> 
		class tquat
		{
		public:
			valType x, y, z, w;

			// Constructors
			tquat();
			explicit tquat(valType const & s, tvec3<valType> const & v);
			explicit tquat(valType const & w, valType const & x, valType const & y, valType const & z);

			// Convertions
			//explicit tquat(valType const & pitch, valType const & yaw, valType const & roll);
			//! pitch, yaw, roll
			explicit tquat(tvec3<valType> const & eulerAngles);
			explicit tquat(tmat3x3<valType> const & m);
			explicit tquat(tmat4x4<valType> const & m);

			// Accesses
			valType& operator[](int i);
			valType operator[](int i) const;

			// Operators
			tquat<valType>& operator*=(valType const & s);
			tquat<valType>& operator/=(valType const & s);
		};

		template <typename valType> 
		detail::tquat<valType> operator- (
			detail::tquat<valType> const & q);

		template <typename valType> 
		detail::tvec3<valType> operator* (
			detail::tquat<valType> const & q, 
			detail::tvec3<valType> const & v);

		template <typename valType> 
		detail::tvec3<valType> operator* (
			detail::tvec3<valType> const & v,
			detail::tquat<valType> const & q);

		template <typename valType> 
		detail::tvec4<valType> operator* (
			detail::tquat<valType> const & q, 
			detail::tvec4<valType> const & v);

		template <typename valType> 
		detail::tvec4<valType> operator* (
			detail::tvec4<valType> const & v,
			detail::tquat<valType> const & q);

		template <typename valType> 
		detail::tquat<valType> operator* (
			detail::tquat<valType> const & q, 
			valType const & s);

		template <typename valType> 
		detail::tquat<valType> operator* (
			valType const & s,
			detail::tquat<valType> const & q);

		template <typename valType> 
		detail::tquat<valType> operator/ (
			detail::tquat<valType> const & q, 
			valType const & s);

	} //namespace detail

	namespace gtc{
	//! GLM_GTC_quaternion extension: Quaternion types and functions
    namespace quaternion
    {
		//! Returns the length of the quaternion x. 
		//! From GLM_GTC_quaternion extension.
        template <typename valType> 
		valType length(
			detail::tquat<valType> const & q);

        //! Returns the normalized quaternion of from x. 
		//! From GLM_GTC_quaternion extension.
		template <typename valType> 
		detail::tquat<valType> normalize(
			detail::tquat<valType> const & q);
		
        //! Returns dot product of q1 and q2, i.e., q1[0] * q2[0] + q1[1] * q2[1] + ... 
		//! From GLM_GTC_quaternion extension.
		template <typename valType> 
		valType dot(
			detail::tquat<valType> const & q1, 
			detail::tquat<valType> const & q2);

        //! Returns the cross product of q1 and q2. 
		//! From GLM_GTC_quaternion extension.
		template <typename valType> 
		detail::tquat<valType> cross(
			detail::tquat<valType> const & q1, 
			detail::tquat<valType> const & q2);
		
		//! Returns a LERP interpolated quaternion of x and y according a. 
		//! From GLM_GTC_quaternion extension.
		template <typename valType> 
		detail::tquat<valType> mix(
			detail::tquat<valType> const & x, 
			detail::tquat<valType> const & y, 
			valType const & a);
		
		//! Returns the q conjugate. 
		//! From GLM_GTC_quaternion extension.
        template <typename valType> 
		detail::tquat<valType> conjugate(
			detail::tquat<valType> const & q);

		//! Returns the q inverse. 
		//! From GLM_GTC_quaternion extension.
        template <typename valType> 
		detail::tquat<valType> inverse(
			detail::tquat<valType> const & q);

		//! Rotates a quaternion from an vector of 3 components axis and an angle expressed in degrees.
		//! From GLM_GTC_quaternion extension.
		template <typename valType> 
		detail::tquat<valType> rotate(
			detail::tquat<valType> const & q, 
			valType const & angle, 
			detail::tvec3<valType> const & v);

		//! Converts a quaternion to a 3 * 3 matrix. 
		//! From GLM_GTC_quaternion extension.
        template <typename valType> 
		detail::tmat3x3<valType> mat3_cast(
			detail::tquat<valType> const & x);

		//! Converts a quaternion to a 4 * 4 matrix. 
		//! From GLM_GTC_quaternion extension.
		template <typename valType> 
		detail::tmat4x4<valType> mat4_cast(
			detail::tquat<valType> const & x);

		//! Converts a 3 * 3 matrix to a quaternion. 
		//! From GLM_GTC_quaternion extension.
		template <typename valType> 
		detail::tquat<valType> quat_cast(
			detail::tmat3x3<valType> const & x);

		//! Converts a 4 * 4 matrix to a quaternion. 
		//! From GLM_GTC_quaternion extension.
		template <typename valType> 
		detail::tquat<valType> quat_cast(
			detail::tmat4x4<valType> const & x);

		//! Quaternion of floating-point numbers. 
		//! From GLM_GTC_quaternion extension.
        typedef detail::tquat<float> quat;

    }//namespace quaternion
    }//namespace gtc
} //namespace glm

#define GLM_GTC_quaternion namespace gtc::quaternion
#ifndef GLM_GTC_GLOBAL
namespace glm {using GLM_GTC_quaternion;}
#endif//GLM_GTC_GLOBAL

#include "quaternion.inl"

#endif//glm_gtc_quaternion
