#include "Precompiled.h"
#include "VarType.h"

namespace Mod
{
	namespace VarType
	{
		bool Convertable( Type to, Type from )
		{
			if( to == from )
				return true;

			switch( to )
			{
			case FLOAT:
				switch( from )
				{
				case FLOAT2:
				case FLOAT3:
				case FLOAT4:
					return true;
				default:
					break;
				}
				break;
			case FLOAT2:
				switch( from )
				{
				case FLOAT3:
				case FLOAT4:
					return true;
				default:
					break;
				}
				break;
			case FLOAT3:
				switch( from )
				{
				case FLOAT4:
					return true;
				default:
					break;
				}
				break;

			}

			return false;
		}

		//------------------------------------------------------------------------

		bool Combatible( Type t1, Type t2 )
		{
			if( t1 == t2 )
				return true;

			if( t1 == UINT && t2 == INT )
				return true;

			if( t1 == UINT2 && t2 == INT2 )
				return true;

			if( t1 == UINT3 && t2 == INT3 )
				return true;

			if( t1 == UINT4 && t2 == INT4 )
				return true;

			if( t1 == UINT_VEC && t2 == INT_VEC )
				return true;

			if( t1 == UINT2_VEC && t2 == INT2_VEC )
				return true;

			if( t1 == UINT3_VEC && t2 == INT3_VEC )
				return true;

			if( t1 == UINT4_VEC && t2 == INT4_VEC )
				return true;

			return false;
		}

		//------------------------------------------------------------------------

		bool IsVector( Type type )
		{

#define MD_TYPE_CASE_ROW(type) case type##_VEC:case type##2_VEC:case type##3_VEC:case type##4_VEC:case type##2x2_VEC:case type##3x3_VEC:case type##4x4_VEC:case type##3x4_VEC:
			switch(type)
			{
			MD_TYPE_CASE_ROW(FLOAT)
			MD_TYPE_CASE_ROW(INT)
			MD_TYPE_CASE_ROW(UINT)
				return true;
			default:
				return false;
			}
#undef MD_TYPE_CASE_ROW
		}
	}

}