///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-04-29
// Updated : 2009-04-29
// Licence : This source is under MIT License
// File    : glm/gtc/matrix_projection.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace gtc{
namespace matrix_projection
{
	template <typename valType> 
	inline detail::tmat4x4<valType> ortho(
		valType const & left, 
		valType const & right, 
		valType const & bottom, 
		valType const & top)
	{
		detail::tmat4x4<valType> Result(1);
		Result[0][0] = valType(2) / (right - left);
		Result[1][1] = valType(2) / (top - bottom);
		Result[2][2] = - valType(1);
		Result[3][0] = - (right + left) / (right - left);
		Result[3][1] = - (top + bottom) / (top - bottom);
		return Result;
	}

	template <typename valType> 
	inline detail::tmat4x4<valType> ortho(
		valType const & left, 
		valType const & right, 
		valType const & bottom, 
		valType const & top, 
		valType const & zNear, 
		valType const & zFar)
	{
		detail::tmat4x4<valType> Result(1);
		Result[0][0] = valType(2) / (right - left);
		Result[1][1] = valType(2) / (top - bottom);
		Result[2][2] = - valType(2) / (zFar - zNear);
		Result[3][0] = - (right + left) / (right - left);
		Result[3][1] = - (top + bottom) / (top - bottom);
		Result[3][2] = - (zFar + zNear) / (zFar - zNear);
		return Result;
	}

	template <typename valType> 
	inline detail::tmat4x4<valType> frustum(
		valType const & left, 
		valType const & right, 
		valType const & bottom, 
		valType const & top, 
		valType const & nearVal, 
		valType const & farVal)
	{
		detail::tmat4x4<valType> Result(0);
		Result[0][0] = (valType(2) * nearVal) / (right - left);
		Result[1][1] = (valType(2) * nearVal) / (top - bottom);
		Result[2][0] = (right + left) / (right - left);
		Result[2][1] = (top + bottom) / (top - bottom);
		Result[2][2] = -(farVal + nearVal) / (farVal - nearVal);
		Result[2][3] = valType(-1);
		Result[3][2] = -(valType(2) * farVal * nearVal) / (farVal - nearVal);
		return Result;
	}

	template <typename valType> 
	inline detail::tmat4x4<valType> perspective(
		valType const & fovy, 
		valType const & aspect, 
		valType const & zNear, 
		valType const & zFar)
	{
		valType range = tan(radians(fovy / valType(2))) * zNear;	
		valType left = -range * aspect;
		valType right = range * aspect;
		valType bottom = -range;
		valType top = range;

		detail::tmat4x4<valType> Result(valType(0));
		Result[0][0] = (valType(2) * zNear) / (right - left);
		Result[1][1] = (valType(2) * zNear) / (top - bottom);
		Result[2][2] = - (zFar + zNear) / (zFar - zNear);
		Result[2][3] = - valType(1);
		Result[3][2] = - (valType(2) * zFar * zNear) / (zFar - zNear);
		return Result;
	}

	template <typename valTypeT, typename valTypeU>
	inline detail::tvec3<valTypeT> project(
		detail::tvec3<valTypeT> const & obj, 
		detail::tmat4x4<valTypeT> const & model, 
		detail::tmat4x4<valTypeT> const & proj, 
		detail::tvec4<valTypeU> const & viewport)
	{
		detail::tvec4<valTypeT> tmp = detail::tvec4<valTypeT>(obj, valTypeT(1));
		tmp = model * tmp;
		tmp = proj * tmp;

		tmp /= tmp.w;
		tmp = tmp * valTypeT(0.5) + valTypeT(0.5);
		tmp[0] = tmp[0] * valTypeT(viewport[2]) + valTypeT(viewport[0]);
		tmp[1] = tmp[1] * valTypeT(viewport[3]) + valTypeT(viewport[1]);

		return detail::tvec3<valTypeT>(tmp);
	}

	template <typename valTypeT, typename valTypeU>
	inline detail::tvec3<valTypeT> unProject(
		detail::tvec3<valTypeT> const & win, 
		detail::tmat4x4<valTypeT> const & model, 
		detail::tmat4x4<valTypeT> const & proj, 
		detail::tvec4<valTypeU> const & viewport)
	{
		detail::tmat4x4<valTypeT> inverse = glm::inverse(proj * model);

		detail::tvec4<valTypeT> tmp = detail::tvec4<valTypeT>(win, valTypeT(1));
		tmp.x = (tmp.x - valTypeT(viewport[0])) / valTypeT(viewport[2]);
		tmp.y = (tmp.y - valTypeT(viewport[1])) / valTypeT(viewport[3]);
		tmp = tmp * valTypeT(2) - valTypeT(1);

		detail::tvec4<valTypeT> obj = inverse * tmp;
		obj /= obj.w;

		return detail::tvec3<valTypeT>(obj);
	}

}//namespace matrix_projection
}//namespace gtc
}//namespace glm
