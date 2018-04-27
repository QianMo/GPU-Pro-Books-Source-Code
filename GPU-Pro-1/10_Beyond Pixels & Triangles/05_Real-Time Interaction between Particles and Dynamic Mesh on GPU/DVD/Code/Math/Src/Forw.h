#ifndef MATH_MATHFORW_H_INCLUDED
#define MATH_MATHFORW_H_INCLUDED

namespace Mod
{
	namespace Math
	{
#define MD_DEF_VEC_TYPE(type) typedef Types< type > :: Vec type##_vec;
#define MD_DEF_TYPE_AND_VEC_TYPE(type) struct type; MD_DEF_VEC_TYPE(type)
#define MD_DEFINE_MATH_FORW_STRUCTS(type)	\
		MD_DEF_TYPE_AND_VEC_TYPE(type##2)	\
		MD_DEF_TYPE_AND_VEC_TYPE(type##3)	\
		MD_DEF_TYPE_AND_VEC_TYPE(type##4)	\
											\
		MD_DEF_TYPE_AND_VEC_TYPE(type##4x4)	\
		MD_DEF_TYPE_AND_VEC_TYPE(type##4x3)	\
		MD_DEF_TYPE_AND_VEC_TYPE(type##4x2)	\
											\
		MD_DEF_TYPE_AND_VEC_TYPE(type##3x4)	\
		MD_DEF_TYPE_AND_VEC_TYPE(type##3x3)	\
		MD_DEF_TYPE_AND_VEC_TYPE(type##3x2)	\
											\
		MD_DEF_TYPE_AND_VEC_TYPE(type##2x4)	\
		MD_DEF_TYPE_AND_VEC_TYPE(type##2x3)	\
		MD_DEF_TYPE_AND_VEC_TYPE(type##2x2)

		MD_DEFINE_MATH_FORW_STRUCTS(float)
		MD_DEFINE_MATH_FORW_STRUCTS(int)
		MD_DEFINE_MATH_FORW_STRUCTS(uint)

#undef MD_DEF_VEC_TYPE
#undef MD_DEFINE_MATH_FORW_STRUCTS

#include "DefDeclareClass.h"

		MOD_DECLARE_BOLD_CLASS(Ray)
		MOD_DECLARE_BOLD_CLASS(BBox)

		MOD_DECLARE_BOLD_STRUCT(Frustum)
		MOD_DECLARE_BOLD_STRUCT(FrustumPoints)

#include "UndefDeclareClass.h"

		typedef Types< BBox > :: Vec BBoxVec;

	}
}

#endif
