#ifndef MATH_TYPES_H_INCLUDED
#define MATH_TYPES_H_INCLUDED

#include "Forw.h"
#include "Half.h"

namespace Mod
{

#define MD_VEC_TYPE(type) type##_vec
#define MD_DEF_VEC_TYPE(type) typedef Types< type > :: Vec MD_VEC_TYPE(type);

#define DEFINE_TUPLE_INDEX_OPERATORS(type, num)							\
			type& operator [] (size_t idx)								\
			{															\
				MD_ASSERT(idx < num);									\
				return elems[idx];										\
			}															\
			const type& operator [] (size_t idx) const					\
			{															\
				MD_ASSERT(idx < num);									\
				return elems[idx];										\
			}															\

#define DEFINE_ARITHMETIC_OPERATORS(type,num)							\
		type& operator += ( const type& val)							\
		{																\
			for( int i = 0; i < num; i ++ )								\
			{															\
				elems[i] += val.elems[i];								\
			}															\
			return *this;												\
		}																\
		type operator - () const										\
		{																\
			type val;													\
			for( int i = 0; i < num; i ++ )								\
			{															\
				val.elems[i] = -elems[i];								\
			}															\
			return val;													\
		}																\
		type& operator -= ( const type& val )							\
		{																\
			*this += -val;												\
			return *this;												\
		}																\
		type& operator *= ( const type& val )							\
		{																\
			for( int i = 0; i < num; i ++ )								\
			{															\
				elems[i] *= val.elems[i];								\
			}															\
			return *this;												\
		}																\
		type& operator /= ( const type& val )							\
		{																\
			for( int i = 0; i < num; i ++ )								\
			{															\
				elems[i] /= val.elems[i];								\
			}															\
			return *this;												\
		}

#define DEFINE_1D_ARITHMETIC_OPERATORS(type,num)						\
		type##num operator * ( type##num::comp_type k ) const			\
		{																\
			type##num res;												\
			for( int i = 0; i < num; i ++ )								\
				res.elems[i] = elems[i]*k;								\
			return res;													\
		}																\
		type##num operator / ( type##num::comp_type k ) const			\
		{																\
			type##num res;												\
			for( int i = 0; i < num; i ++ )								\
				res.elems[i] = elems[i]/k;								\
			return res;													\
		}																\
		type##num& operator *= ( type##num::comp_type k )				\
		{																\
			for( int i = 0; i < num; i ++ )								\
				elems[i] *= k;											\
			return *this;												\
		}																\
		type##num& operator /= ( type##num::comp_type k )				\
		{																\
			for( int i = 0; i < num; i ++ )								\
				elems[i] /= k;											\
			return *this;												\
		}


#define DEFINE_FREE_ARITHMETIC_OPERATORS(type,num)							\
		inline																\
		type##num operator + ( const type##num& v1, const type##num& v2 )	\
		{																	\
			type##num res(v1);												\
			res += v2;														\
			return res;														\
		}																	\
		inline																\
		type##num operator - ( const type##num& v1, const type##num& v2 )	\
		{																	\
			type##num res(v1);												\
			res -= v2;														\
			return res;														\
		}																	\
		inline																\
		type##num operator * ( const type##num& v1, const type##num& v2 )	\
		{																	\
			type##num res(v1);												\
			res *= v2;														\
			return res;														\
		}
	
#define DEFINE_VEC_FREE_ARITHMETIC_OPERATORS(type)							\
		inline																\
		type operator + ( const type& val1, const type& val2 )				\
		{																	\
			MD_ASSERT( val1.size() == val2.size() );						\
			type res( val1.size() );										\
			for( size_t i = 0, e = val1.size(); i < e; i ++ )				\
				res[i] = val1[i] + val2[i];									\
			return res;														\
		}																	\
		inline																\
		type operator * ( const type& val1, const type& val2 )				\
		{																	\
			MD_ASSERT( val1.size() == val2.size() );						\
			type res( val1.size() );										\
			for( size_t i = 0, e = val1.size(); i < e; i ++ )				\
				res[i] = val1[i] * val2[i];									\
			return res;														\
		}																	\
		inline																\
		type operator * ( const type& val1, const type::value_type& val2 )	\
		{																	\
			type res( val1.size() );										\
			for( size_t i = 0, e = val1.size(); i < e; i ++ )				\
				res[i] = val1[i] * val2;									\
			return res;														\
		}																	\
		inline																\
		type operator * ( const type::value_type& val1, const type& val2 )	\
		{																	\
			type res( val2.size() );										\
			for( size_t i = 0, e = val2.size(); i < e; i ++ )				\
				res[i] = val2[i] * val1;									\
			return res;														\
		}


#define DEFINE_1D_FREE_ARITHMETIC_OPERATORS(type,num)						\
		inline																\
		type##num operator * ( type k, const type##num& val )				\
		{																	\
			return val*k;													\
		}																	\
		DEFINE_FREE_ARITHMETIC_OPERATORS(type,num)							\
		DEFINE_VEC_FREE_ARITHMETIC_OPERATORS(MD_VEC_TYPE(type##num))



#define DEFINE_MEMBER_OPERATORS(type,num)			\
		DEFINE_TUPLE_INDEX_OPERATORS(type, num)		\
		DEFINE_ARITHMETIC_OPERATORS(type##num, num)

#define DEFINE_TUPLE_STRUCTS_OF_TYPE(type)								\
		struct type##2													\
		{																\
			typedef type comp_type;										\
			static const UINT32 COMPONENT_COUNT = 2;					\
			union														\
			{															\
				struct													\
				{														\
					type x, y;											\
				};														\
				struct													\
				{														\
					type r, g;											\
				};														\
				type elems[2];											\
			};															\
			type##2() {}												\
			type##2( type x, type y ) : x(x), y(y) {}					\
			DEFINE_MEMBER_OPERATORS(type,2)								\
			DEFINE_1D_ARITHMETIC_OPERATORS(type,2)						\
		};																\
		MD_DEF_VEC_TYPE(type##2)										\
		DEFINE_1D_FREE_ARITHMETIC_OPERATORS(type,2)						\
		struct type##3													\
		{																\
			typedef type comp_type;										\
			static const UINT32 COMPONENT_COUNT = 3;					\
			union														\
			{															\
				struct													\
				{														\
					type x, y, z;										\
				};														\
				struct													\
				{														\
					type r, g, b;										\
				};														\
				type elems[3];											\
			};															\
			type##3() {}												\
			type##3( type x, type y, type z ) : x(x), y(y), z(z) {}		\
			operator type##2 () const { return type##2( x, y ); }		\
			DEFINE_MEMBER_OPERATORS(type,3)								\
			DEFINE_1D_ARITHMETIC_OPERATORS(type,3)						\
		};																\
		MD_DEF_VEC_TYPE(type##3)										\
		DEFINE_1D_FREE_ARITHMETIC_OPERATORS(type,3)						\
		struct type##4													\
		{																\
			typedef type comp_type;										\
			static const UINT32 COMPONENT_COUNT = 4;					\
			union														\
			{															\
				struct													\
				{														\
					type x, y, z, w;									\
				};														\
				struct													\
				{														\
					type r, g, b, a;									\
				};														\
				type elems[4];											\
			};															\
			type##4() {}												\
			type##4( type x, type y, type z, type w) :					\
			x(x), y(y), z(z), w(w) {}									\
			type##4( type##3 val, type w) :								\
			x( val.x ), y( val.y ), z( val.z ), w(w) {}					\
			DEFINE_MEMBER_OPERATORS(type,4)								\
			DEFINE_1D_ARITHMETIC_OPERATORS(type,4)						\
			operator type##3 () const { return type##3( x, y, z ); }	\
			operator type##2 () const { return type##2( x, y ); }		\
		};																\
		MD_DEF_VEC_TYPE(type##4)										\
		DEFINE_1D_FREE_ARITHMETIC_OPERATORS(type,4)


#define DEFINE_4x4_COMPS(type)											\
			type##4x4(	type a11, type a21, type a31, type a41,			\
						type a12, type a22, type a32, type a42,			\
						type a13, type a23, type a33, type a43,			\
						type a14, type a24, type a34, type a44	)		\
			{															\
				elems[0] = type##4( a11, a21, a31, a41 );				\
				elems[1] = type##4( a12, a22, a32, a42 );				\
				elems[2] = type##4( a13, a23, a33, a43 );				\
				elems[3] = type##4( a14, a24, a34, a44 );				\
			}

#define DEFINE_4x3_COMPS(type)										\
			type##4x3(	type a11, type a21, type a31, type a41,		\
						type a12, type a22, type a32, type a42,		\
						type a13, type a23, type a33, type a43	)	\
			{														\
				elems[0] = type##4( a11, a21, a31, a41 );			\
				elems[1] = type##4( a12, a22, a32, a42 );			\
				elems[2] = type##4( a13, a23, a33, a43 );			\
			}

#define DEFINE_4x2_COMPS(type)										\
			type##4x2(	type a11, type a21, type a31, type a41,		\
						type a12, type a22, type a32, type a42	)	\
			{														\
				elems[0] = type##4( a11, a21, a31, a41 );			\
				elems[1] = type##4( a12, a22, a32, a42 );			\
			}

#define DEFINE_3x4_COMPS(type)								\
			type##3x4(	type a11, type a21, type a31, 		\
						type a12, type a22, type a32, 		\
						type a13, type a23, type a33, 		\
						type a14, type a24, type a34 )		\
			{												\
				elems[0] = type##3( a11, a21, a31 );		\
				elems[1] = type##3( a12, a22, a32 );		\
				elems[2] = type##3( a13, a23, a33 );		\
				elems[3] = type##3( a14, a24, a34 );		\
			}

#define DEFINE_3x3_COMPS(type)							\
			type##3x3(	type a11, type a21, type a31,	\
						type a12, type a22, type a32,	\
						type a13, type a23, type a33)	\
			{											\
				elems[0] = type##3( a11, a21, a31 );	\
				elems[1] = type##3( a12, a22, a32 );	\
				elems[2] = type##3( a13, a23, a33 );	\
			}

#define DEFINE_3x2_COMPS(type)							\
			type##3x2(	type a11, type a21, type a31, 	\
						type a12, type a22, type a32)	\
			{											\
				elems[0] = type##3( a11, a21, a31 );	\
				elems[1] = type##3( a12, a22, a32 );	\
			}

#define DEFINE_2x4_COMPS(type)						\
			type##2x4(	type a11, type a21,			\
						type a12, type a22,			\
						type a13, type a23,			\
						type a14, type a24 )		\
			{										\
				elems[0] = type##2( a11, a21 );		\
				elems[1] = type##2( a12, a22 );		\
				elems[2] = type##2( a13, a23 );		\
				elems[3] = type##2( a14, a24 );		\
			}

#define DEFINE_2x3_COMPS(type)					\
			type##2x3(	type a11, type a21,		\
						type a12, type a22,		\
						type a13, type a23 )	\
			{									\
				elems[0] = type##2( a11, a21 );	\
				elems[1] = type##2( a12, a22 );	\
				elems[2] = type##2( a13, a23 );	\
			}

#define DEFINE_2x2_COMPS(type)					\
			type##2x2(	type a11, type a21,		\
						type a12, type a22)		\
			{									\
				elems[0] = type##2( a11, a21 );	\
				elems[1] = type##2( a12, a22 );	\
			}


#define DEFINE_2D_TUPLE_STRUCTS_OF_TYPE(type,num,COMPS)					\
		struct type##x##num												\
		{																\
			typedef type comp_type;										\
			type elems[num];											\
			type##x##num() {}											\
			COMPS														\
			DEFINE_TUPLE_INDEX_OPERATORS(type, num)						\
			DEFINE_ARITHMETIC_OPERATORS(type##x##num, num)				\
		};																\
		DEFINE_FREE_ARITHMETIC_OPERATORS(type##x,num)					\
		MD_DEF_VEC_TYPE(type##x##num)									\
		DEFINE_VEC_FREE_ARITHMETIC_OPERATORS(MD_VEC_TYPE(type##x##num))

#define DEFINE_TUPLE_ARRAYS_OF_TYPE(type)									\
		DEFINE_2D_TUPLE_STRUCTS_OF_TYPE(type##4,4,DEFINE_4x4_COMPS(type))	\
		DEFINE_2D_TUPLE_STRUCTS_OF_TYPE(type##3,4,DEFINE_3x4_COMPS(type))	\
		DEFINE_2D_TUPLE_STRUCTS_OF_TYPE(type##2,4,DEFINE_2x4_COMPS(type))	\
																			\
		DEFINE_2D_TUPLE_STRUCTS_OF_TYPE(type##4,3,DEFINE_4x3_COMPS(type))	\
		DEFINE_2D_TUPLE_STRUCTS_OF_TYPE(type##3,3,DEFINE_3x3_COMPS(type))	\
		DEFINE_2D_TUPLE_STRUCTS_OF_TYPE(type##2,3,DEFINE_2x3_COMPS(type))	\
																			\
		DEFINE_2D_TUPLE_STRUCTS_OF_TYPE(type##4,2,DEFINE_4x2_COMPS(type))	\
		DEFINE_2D_TUPLE_STRUCTS_OF_TYPE(type##3,2,DEFINE_3x2_COMPS(type))	\
		DEFINE_2D_TUPLE_STRUCTS_OF_TYPE(type##2,2,DEFINE_2x2_COMPS(type))

#define DEFINE_ALL_TUPPLE_TYPES(type)			\
		DEFINE_TUPLE_STRUCTS_OF_TYPE(type)		\
		DEFINE_TUPLE_ARRAYS_OF_TYPE(type)		\
		MD_DEF_VEC_TYPE(type)

#pragma warning(push)
#pragma warning(disable : 4201)

namespace Math
{
#ifdef _MSC_VER
	typedef unsigned __int32	uint;
	typedef unsigned __int16	ushort;
	typedef __int8				sbyte;
	typedef unsigned __int8		ubyte;
#else
#error Compiler not supported...
#endif

#pragma warning(push)
#pragma warning(disable:4146)

	DEFINE_ALL_TUPPLE_TYPES(half)
	DEFINE_ALL_TUPPLE_TYPES(float)
	DEFINE_ALL_TUPPLE_TYPES(int)
	DEFINE_ALL_TUPPLE_TYPES(uint)
	DEFINE_ALL_TUPPLE_TYPES(short)
	DEFINE_ALL_TUPPLE_TYPES(ushort)
	DEFINE_ALL_TUPPLE_TYPES(sbyte)
	DEFINE_ALL_TUPPLE_TYPES(ubyte)

#pragma warning(pop)

}

#pragma warning(pop)

}

#undef DEFINE_TUPLE_INDEX_OPERATORS
#undef DEFINE_ARITHMETIC_OPERATORS
#undef DEFINE_MEMBER_OPERATORS
#undef DEFINE_TUPLE_STRUCTS_OF_TYPE
#undef DEFINE_2D_TUPLE_STRUCTS_OF_TYPE
#undef DEFINE_TUPLE_ARRAYS_OF_TYPE
#undef DEFINE_ALL_TUPPLE_TYPES
#undef DEFINE_1D_ARITHMETIC_OPERATORS
#undef DEFINE_FREE_ARITHMETIC_OPERATORS
#undef DEFINE_1D_FREE_ARITHMETIC_OPERATORS
#undef DEFINE_VEC_FREE_ARITHMETIC_OPERATORS
#undef DEFINE_4x4_CONSTRUCTOR
#undef DEFINE_4x3_CONSTRUCTOR
#undef DEFINE_4x2_CONSTRUCTOR
#undef DEFINE_3x4_CONSTRUCTOR
#undef DEFINE_3x3_CONSTRUCTOR
#undef DEFINE_3x2_CONSTRUCTOR
#undef DEFINE_2x4_CONSTRUCTOR
#undef DEFINE_2x3_CONSTRUCTOR
#undef DEFINE_2x2_CONSTRUCTOR
#undef MD_DEF_VEC_TYPE
#undef MD_VEC_TYPE

#endif