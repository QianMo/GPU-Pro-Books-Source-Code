#ifndef COMMON_VARTYPE_H_INCLUDED
#define COMMON_VARTYPE_H_INCLUDED

#include "Math/Src/Types.h"

namespace Mod
{
	namespace VarType
	{
		using namespace Math;
#define MD_VARBIND_TYPE_LINE(name)	name,		name##2,		name##3,		name##4,		name##2x2,		name##3x3,		name##4x4,		name##3x4,		\
									name##_VEC,	name##2_VEC,	name##3_VEC,	name##4_VEC,	name##2x2_VEC,	name##3x3_VEC,	name##4x4_VEC,	name##3x4_VEC,
		enum Type
		{
			MD_VARBIND_TYPE_LINE( FLOAT )
			MD_VARBIND_TYPE_LINE( INT )
			MD_VARBIND_TYPE_LINE( UINT )
			SHADER_RESOURCE,
			CONSTANT_BUFFER,
			UNKNOWN = -1,
			NUM_TYPES = CONSTANT_BUFFER + 2
		};
#undef MD_VARBIND_TYPE_LINE

		template <typename T>
		struct TypeToEnum
		{
			static const Type Result = UNKNOWN;
		};

		template<Type E>
		struct EnumToType
		{

		};

#define MD_DEFINE_TYPE_TO_ENUM_LINK(type,name)		\
		template <>									\
		struct TypeToEnum<type>						\
		{											\
			static const Type Result = name;		\
		};											\
		template <>									\
		struct EnumToType<name>						\
		{											\
			typedef type Result;					\
		};

#define MD_DEFINE_TYPE_VEC_TO_ENUM_LINK(type,name)			\
		MD_DEFINE_TYPE_TO_ENUM_LINK(type,name)				\
		MD_DEFINE_TYPE_TO_ENUM_LINK(type##_vec,name##_VEC)

		

#define MD_DEFINE_TYPE_TO_ENUM_LINK_ROW(type,name)				\
		MD_DEFINE_TYPE_VEC_TO_ENUM_LINK(type##2,name##2)		\
		MD_DEFINE_TYPE_VEC_TO_ENUM_LINK(type##3,name##3)		\
		MD_DEFINE_TYPE_VEC_TO_ENUM_LINK(type##4,name##4)		\
		MD_DEFINE_TYPE_VEC_TO_ENUM_LINK(type##2x2,name##2x2)	\
		MD_DEFINE_TYPE_VEC_TO_ENUM_LINK(type##3x3,name##3x3)	\
		MD_DEFINE_TYPE_VEC_TO_ENUM_LINK(type##4x4,name##4x4)	\
		MD_DEFINE_TYPE_VEC_TO_ENUM_LINK(type##3x4,name##3x4)

		MD_DEFINE_TYPE_VEC_TO_ENUM_LINK(float,FLOAT)
		MD_DEFINE_TYPE_TO_ENUM_LINK_ROW(float,FLOAT)
		MD_DEFINE_TYPE_VEC_TO_ENUM_LINK(int,INT)
		MD_DEFINE_TYPE_TO_ENUM_LINK_ROW(int,INT)
		MD_DEFINE_TYPE_VEC_TO_ENUM_LINK(uint,UINT)
		MD_DEFINE_TYPE_TO_ENUM_LINK_ROW(uint,UINT)

#undef MD_DEFINE_TYPE_TO_ENUM_LINK_ROW
#undef MD_DEFINE_TYPE_TO_ENUM_LINK
#undef MD_DEFINE_TYPE_VEC_TO_ENUM_LINK
		bool Convertable( Type to, Type from );
		bool Combatible( Type t1, Type t2 );
		bool IsVector( Type type );
	}
}

#endif