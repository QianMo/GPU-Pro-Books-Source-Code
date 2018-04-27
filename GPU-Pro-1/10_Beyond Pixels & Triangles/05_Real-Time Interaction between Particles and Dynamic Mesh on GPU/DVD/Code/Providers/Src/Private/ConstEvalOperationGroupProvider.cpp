#include "Precompiled.h"

#include "Math/Src/Operations.h"
#include "Common/Src/VarType.h"

#include "ConstEvalNode.h"
#include "ConstEvalOperationGroup.h"

#include "ConstEvalOperationGroupProviderConfig.h"
#include "ConstEvalOperationGroupProvider.h"

#define MD_NAMESPACE ConstEvalOperationGroupProviderNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	template class ConstEvalOperationGroupProviderNS::ConfigurableImpl< ConstEvalOperationGroupProviderConfig >;

	typedef ConstEvalNode::Parents Parents;
	typedef ConstEvalNode::Children Children;
	using namespace Math;

	namespace
	{
		template <typename R, typename T1, typename T2>
		void mulfunc_n_n( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 2 );
			res.Set( (R)mul( children[0]->GetVal<T1>(), children[1]->GetVal<T2>() ) );
		}

		void get_scale( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 1 );

			const float3x4& val = children[0]->GetVal<float3x4>();

			float3 s,t; float4 r;
			m3x4Decompose( val, t, r, s );

			res.Set( s );
		}

		void make_scale( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 1 );

			const float3& val = children[0]->GetVal<float3>();
			res.Set( m3x4Scale( val.x, val.y, val.z ) );
		}

		template <typename T>
		void inverse_nxn( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 1 );
			res.Set( inverse( children[0]->GetVal<T>() ) );
		}

		template <typename T>
		void normalize_n( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 1 );
			res.Set( normalize( children[0]->GetVal<T>() ) );
		}


		void conv_3_1_to_4_f( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 2 );
			float3 val = children[0]->GetVal<float3>();
			res.Set( float4( val.x, val.y, val.z, children[1]->GetVal<float>() ) );
		}

		void conv_3v_1_to_4v_f( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 2 );
			const float3_vec& val1 = children[0]->GetVal<float3_vec>();
			float val2 = children[1]->GetVal<float>();
			float4_vec res_val( val1.size() );

			for( UINT32 i = 0, e = (UINT32)val1.size(); i < e; i++ )
			{
				res_val[ i ] = float4( val1[ i ], val2 );
			}

			res.Set( res_val );
		}

		template <typename T>
		void conv_nx1_to_n( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == T::COMPONENT_COUNT );

			T resval;

			for( UINT32 i = 0; i < T::COMPONENT_COUNT; i ++ )
			{
				resval[i] = children[i]->GetVal<T::comp_type>();
			}

			res.Set( resval );
		}

		void conv_16_to_4x4_f( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 16 );

			float4x4 res_mat;

			for( UINT32 i = 0, k = 0; i < 4; i ++ )
			{
				for( UINT32 j = 0; j < 4; j ++, k ++  )
				{
					res_mat[i][j] = children[k]->GetVal<float>();
				}
			}

			res.Set( res_mat );
		}

		void conv_12_to_3x4_f( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 12 );

			float3x4 res_mat;

			for( UINT32 i = 0, k = 0; i < 4; i ++ )
			{
				for( UINT32 j = 0; j < 3; j ++, k ++  )
				{
					res_mat[i][j] = children[k]->GetVal<float>();
				}
			}

			res.Set( res_mat );
		}

		void conv_4_to_3_f( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 1 );

			float3 val = children[0]->GetVal<float4>();
			res.Set( val );
		}

		void conv_4vec_to_3vec_f( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 1 );
			const float4_vec& val = children[0]->GetVal<float4_vec>();

			float3_vec resval( val.size() );

			for( size_t i = 0, e = val.size(); i < e; i ++ )
			{
				resval[ i ] = val[ i ];
			}

			res.Set( resval );
		}

		template <typename T>
		void add_n_to_n( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 2 );
			T a = children[0]->GetVal<T>();
			T b = children[1]->GetVal<T>();

			res.Set( a + b );
		}

		template <typename V, typename T, bool F >
		void add_n_to_vec( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 2 );
			V a = children[F]->GetVal<V>();
			T b = children[!F]->GetVal<T>();

			for( size_t i = 0, e = a.size(); i < e; i ++ )
			{
				a[i] += b;
			}

			res.Set( a );
		}

		template <typename T>
		void sub_n_from_n( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 2 );
			T a = children[0]->GetVal<T>();
			T b = children[1]->GetVal<T>();

			res.Set( a - b );
		}

		template <typename T1, typename T2>
		void mul_t_x_t( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 2 );
			T1 a = children[0]->GetVal<T1>();
			T2 b = children[1]->GetVal<T2>();

			res.Set( a * b );
		}

		template <typename T>
		void div_t_by_t( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 2 );
			T a = children[0]->GetVal<T>();
			T b = children[1]->GetVal<T>();

			res.Set( a / b );
		}

		template <typename T>
		void div_n_by_n( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 2 );
			T a = children[0]->GetVal<T>();
			T b = children[1]->GetVal<T>();

			T resval;

			for( UINT32 i = 0; i < T::COMPONENT_COUNT; i ++ )
			{
				resval.elems[ i ] = a[ i ] / b[ i ];
			}

			res.Set( resval );
		}

		template <typename T1, typename T2>
		void div_n_by_1( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 2 );
			T1 a = children[0]->GetVal<T1>();
			T2 b = children[1]->GetVal<T2>();

			res.Set( a / b );
		}

		template <typename T>
		void unary_minus( VarVariant& res, Children& children )
		{
			MD_FERROR_ON_FALSE( children.size() == 1 );
			res.Set( -children[0]->GetVal<T>() );
		}

		template <typename T>
		void max_t( VarVariant& res, Children& children )
		{
			using std::max;

			MD_FERROR_ON_FALSE( children.size() == 2 );
			res.Set( max( children[0]->GetVal<T>(), children[1]->GetVal<T>() ) );
		}

		template <typename T>
		void min_t( VarVariant& res, Children& children )
		{
			using std::min;

			MD_FERROR_ON_FALSE( children.size() == 2 );
			res.Set( min( children[0]->GetVal<T>(), children[1]->GetVal<T>() ) );
		}
	}

	//------------------------------------------------------------------------

	EXP_IMP
	ConstEvalOperationGroupProvider::ConstEvalOperationGroupProvider( const ConstEvalOperationGroupProviderConfig& cfg ) :
	Parent( cfg )
	{

		typedef ConstEvalOperationGroup::Operations Operations;

#define MD_ADD_ITEM(name) AddItem( L##name, ConstEvalOperationGroupPtr ( new ConstEvalOperationGroup( L##name, ops ) ) )

		// mul
		{
			Operations ops;
			//												result_type		type1			type2				result_type				type1					type2
			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float4x4,		float4x4,		float4x4		>,	VarType::FLOAT4x4,		VarType::FLOAT4x4,		VarType::FLOAT4x4		) );
			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float4x4,		float3x4,		float4x4		>,	VarType::FLOAT4x4,		VarType::FLOAT3x4,		VarType::FLOAT4x4		) );
			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float4x4,		float4x4,		float3x4		>,	VarType::FLOAT4x4,		VarType::FLOAT4x4,		VarType::FLOAT3x4		) );
			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float3x4,		float3x4,		float3x4		>,	VarType::FLOAT3x4,		VarType::FLOAT3x4,		VarType::FLOAT3x4		) );

			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float3,			float3,			float4x4		>,	VarType::FLOAT3,		VarType::FLOAT3,		VarType::FLOAT4x4		) );
			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float4,			float4,			float4x4		>,	VarType::FLOAT4,		VarType::FLOAT4,		VarType::FLOAT4x4		) );
			
			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float3,			float3,			float3x4		>,	VarType::FLOAT3,		VarType::FLOAT3,		VarType::FLOAT3x4		) );
			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float4,			float4,			float3x4		>,	VarType::FLOAT4,		VarType::FLOAT4,		VarType::FLOAT3x4		) );

			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float4x4_vec,	float4x4_vec,	float4x4		>,	VarType::FLOAT4x4_VEC,	VarType::FLOAT4x4_VEC,	VarType::FLOAT4x4		) );
			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float4x4_vec,	float4x4,		float4x4_vec	>,	VarType::FLOAT4x4_VEC,	VarType::FLOAT4x4,		VarType::FLOAT4x4_VEC	) );
			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float4x4_vec,	float4x4,		float3x4_vec	>,	VarType::FLOAT4x4_VEC,	VarType::FLOAT4x4,		VarType::FLOAT3x4_VEC	) );

			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float3x4_vec,	float3x4_vec,	float3x4		>,	VarType::FLOAT3x4_VEC,	VarType::FLOAT3x4_VEC,	VarType::FLOAT3x4		) );
			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float3x4_vec,	float3x4,		float3x4_vec	>,	VarType::FLOAT3x4_VEC,	VarType::FLOAT3x4,		VarType::FLOAT3x4_VEC	) );

			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float4x4_vec,	float3x4_vec,	float4x4		>,	VarType::FLOAT4x4_VEC,	VarType::FLOAT3x4_VEC,	VarType::FLOAT4x4		) );

			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float4_vec,		float4_vec,		float4x4		>,	VarType::FLOAT4_VEC,	VarType::FLOAT4_VEC,	VarType::FLOAT4x4		) );
			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float4_vec,		float3_vec,		float4x4		>,	VarType::FLOAT4_VEC,	VarType::FLOAT3_VEC,	VarType::FLOAT4x4		) );

			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float4_vec,		float4_vec,		float3x4		>,	VarType::FLOAT4_VEC,	VarType::FLOAT4_VEC,	VarType::FLOAT3x4		) );
			ops.push_back( ConstEvalOperation( mulfunc_n_n<	float4_vec,		float3_vec,		float3x4		>,	VarType::FLOAT4_VEC,	VarType::FLOAT3_VEC,	VarType::FLOAT3x4		) );

			MD_ADD_ITEM("mul(");
		}

		// scale
		{
			Operations ops;
			ops.push_back( ConstEvalOperation( get_scale,					VarType::FLOAT3,		VarType::FLOAT3x4			) );

			MD_ADD_ITEM("get_scale(");			
		}

		// scale
		{
			Operations ops;
			ops.push_back( ConstEvalOperation( make_scale,					VarType::FLOAT3x4,		VarType::FLOAT3				) );

			MD_ADD_ITEM("scale(");			
		}

		// inverse
		{
			Operations ops;
			ops.push_back( ConstEvalOperation( inverse_nxn<float4x4>,		VarType::FLOAT4x4,		VarType::FLOAT4x4			) );
			ops.push_back( ConstEvalOperation( inverse_nxn<float4x4_vec>,	VarType::FLOAT4x4_VEC,	VarType::FLOAT4x4_VEC		) );

			ops.push_back( ConstEvalOperation( inverse_nxn<float3x4>,		VarType::FLOAT3x4,		VarType::FLOAT3x4			) );
			ops.push_back( ConstEvalOperation( inverse_nxn<float3x4_vec>,	VarType::FLOAT3x4_VEC,	VarType::FLOAT3x4_VEC		) );

			MD_ADD_ITEM("inverse(");
		}

		// normalize
		{
			Operations ops;
			ops.push_back( ConstEvalOperation( normalize_n<float4>,		VarType::FLOAT4,		VarType::FLOAT4	) );
			ops.push_back( ConstEvalOperation( normalize_n<float3>,		VarType::FLOAT3,		VarType::FLOAT3	) );
			ops.push_back( ConstEvalOperation( normalize_n<float2>,		VarType::FLOAT2,		VarType::FLOAT2	) );

			ops.push_back( ConstEvalOperation( normalize_n<float4_vec>,	VarType::FLOAT4_VEC,	VarType::FLOAT4_VEC	) );
			ops.push_back( ConstEvalOperation( normalize_n<float3_vec>,	VarType::FLOAT3_VEC,	VarType::FLOAT3_VEC	) );
			ops.push_back( ConstEvalOperation( normalize_n<float2_vec>,	VarType::FLOAT2_VEC,	VarType::FLOAT2_VEC	) );

			MD_ADD_ITEM("normalize(");
		}

		// float3
		{
			ConstEvalOperation::VarTypes float_types( 3, VarType::FLOAT );

			Operations ops;
			ops.push_back( ConstEvalOperation( conv_nx1_to_n<float3>, VarType::FLOAT3, float_types ) );

			MD_ADD_ITEM("float3(");
		}

		// float4
		{
			ConstEvalOperation::VarTypes float_types( 4, VarType::FLOAT );
			
			Operations ops;
			ops.push_back( ConstEvalOperation( conv_3_1_to_4_f,			VarType::FLOAT4, VarType::FLOAT3, VarType::FLOAT	) );
			ops.push_back( ConstEvalOperation( conv_nx1_to_n<float4>,	VarType::FLOAT4, float_types						) );

			MD_ADD_ITEM("float4(");
		}

		// float4_vec
		{
			Operations ops;
			ops.push_back( ConstEvalOperation( conv_3v_1_to_4v_f,	VarType::FLOAT4_VEC,	VarType::FLOAT3_VEC,	VarType::FLOAT	) );

			MD_ADD_ITEM("float4_vec(");
		}

		// float4x4
		{
			ConstEvalOperation::VarTypes float_types( 16, VarType::FLOAT );

			Operations ops;
			ops.push_back( ConstEvalOperation( conv_16_to_4x4_f,	VarType::FLOAT4x4, float_types ) );

			MD_ADD_ITEM("float4x4(");
		}

		// float3x4
		{
			ConstEvalOperation::VarTypes float_types( 12, VarType::FLOAT );

			Operations ops;
			ops.push_back( ConstEvalOperation( conv_12_to_3x4_f,	VarType::FLOAT3x4, float_types ) );

			MD_ADD_ITEM("float3x4(");
		}

		// + 
		{
			Operations ops;
			ops.push_back( ConstEvalOperation( add_n_to_n<float>,						VarType::FLOAT,			VarType::FLOAT,			VarType::FLOAT			) );
			ops.push_back( ConstEvalOperation( add_n_to_n<float2>,						VarType::FLOAT2,		VarType::FLOAT2,		VarType::FLOAT2			) );
			ops.push_back( ConstEvalOperation( add_n_to_n<float3>,						VarType::FLOAT3,		VarType::FLOAT3,		VarType::FLOAT3			) );
			ops.push_back( ConstEvalOperation( add_n_to_n<float4>,						VarType::FLOAT4,		VarType::FLOAT4,		VarType::FLOAT4			) );
			ops.push_back( ConstEvalOperation( add_n_to_n<float2x2>,					VarType::FLOAT2x2,		VarType::FLOAT2x2,		VarType::FLOAT2x2		) );
			ops.push_back( ConstEvalOperation( add_n_to_n<float3x3>,					VarType::FLOAT3x3,		VarType::FLOAT3x3,		VarType::FLOAT3x3		) );
			ops.push_back( ConstEvalOperation( add_n_to_n<float4x4>,					VarType::FLOAT4x4,		VarType::FLOAT4x4,		VarType::FLOAT4x4		) );
			ops.push_back( ConstEvalOperation( add_n_to_n<float4x4_vec>,				VarType::FLOAT4x4_VEC,	VarType::FLOAT4x4_VEC,	VarType::FLOAT4x4_VEC	) );
			ops.push_back( ConstEvalOperation( add_n_to_vec<float4x4_vec,float4x4,0>,	VarType::FLOAT4x4_VEC,	VarType::FLOAT4x4_VEC,	VarType::FLOAT4x4		) );
			ops.push_back( ConstEvalOperation( add_n_to_vec<float4x4_vec,float4x4,1>,	VarType::FLOAT4x4_VEC,	VarType::FLOAT4x4,		VarType::FLOAT4x4_VEC	) );

			MD_ADD_ITEM("+");
		}

		// - 
		{
			Operations ops;
			ops.push_back( ConstEvalOperation( sub_n_from_n<float>,		VarType::FLOAT,		VarType::FLOAT,		VarType::FLOAT	) );
			ops.push_back( ConstEvalOperation( sub_n_from_n<float2>,	VarType::FLOAT2,	VarType::FLOAT2,	VarType::FLOAT2	) );
			ops.push_back( ConstEvalOperation( sub_n_from_n<float3>,	VarType::FLOAT3,	VarType::FLOAT3,	VarType::FLOAT3	) );
			ops.push_back( ConstEvalOperation( sub_n_from_n<float4>,	VarType::FLOAT4,	VarType::FLOAT4,	VarType::FLOAT4	) );

			ops.push_back( ConstEvalOperation( unary_minus<float>,		VarType::FLOAT,		VarType::FLOAT	) );
			ops.push_back( ConstEvalOperation( unary_minus<float2>,		VarType::FLOAT2,	VarType::FLOAT2	) );
			ops.push_back( ConstEvalOperation( unary_minus<float3>,		VarType::FLOAT3,	VarType::FLOAT3	) );
			ops.push_back( ConstEvalOperation( unary_minus<float4>,		VarType::FLOAT4,	VarType::FLOAT4	) );


			MD_ADD_ITEM("-");
		}

		// *
		{
			Operations ops;
			ops.push_back( ConstEvalOperation( mul_t_x_t<float,float>,			VarType::FLOAT,			VarType::FLOAT,			VarType::FLOAT			) );

			ops.push_back( ConstEvalOperation( mul_t_x_t<float,float2>,			VarType::FLOAT2,		VarType::FLOAT,			VarType::FLOAT2			) );
			ops.push_back( ConstEvalOperation( mul_t_x_t<float,float3>,			VarType::FLOAT3,		VarType::FLOAT,			VarType::FLOAT3			) );
			ops.push_back( ConstEvalOperation( mul_t_x_t<float,float4>,			VarType::FLOAT4,		VarType::FLOAT,			VarType::FLOAT4			) );
			ops.push_back( ConstEvalOperation( mul_t_x_t<float2,float>,			VarType::FLOAT2,		VarType::FLOAT2,		VarType::FLOAT			) );
			ops.push_back( ConstEvalOperation( mul_t_x_t<float3,float>,			VarType::FLOAT3,		VarType::FLOAT3,		VarType::FLOAT			) );
			ops.push_back( ConstEvalOperation( mul_t_x_t<float4,float>,			VarType::FLOAT4,		VarType::FLOAT4,		VarType::FLOAT			) );

			ops.push_back( ConstEvalOperation( mul_t_x_t<float2_vec,float2>,	VarType::FLOAT2_VEC,	VarType::FLOAT2_VEC,	VarType::FLOAT2			) );
			ops.push_back( ConstEvalOperation( mul_t_x_t<float3_vec,float3>,	VarType::FLOAT3_VEC,	VarType::FLOAT3_VEC,	VarType::FLOAT3			) );
			ops.push_back( ConstEvalOperation( mul_t_x_t<float4_vec,float4>,	VarType::FLOAT4_VEC,	VarType::FLOAT4_VEC,	VarType::FLOAT4			) );
			ops.push_back( ConstEvalOperation( mul_t_x_t<float2,float2_vec>,	VarType::FLOAT2_VEC,	VarType::FLOAT2,		VarType::FLOAT2_VEC		) );
			ops.push_back( ConstEvalOperation( mul_t_x_t<float3,float3_vec>,	VarType::FLOAT3_VEC,	VarType::FLOAT3,		VarType::FLOAT3_VEC		) );
			ops.push_back( ConstEvalOperation( mul_t_x_t<float4,float4_vec>,	VarType::FLOAT4_VEC,	VarType::FLOAT4,		VarType::FLOAT4_VEC		) );
			MD_ADD_ITEM("*");
		}

		// /
		{
			Operations ops;
			ops.push_back( ConstEvalOperation( div_t_by_t<float>,			VarType::FLOAT,		VarType::FLOAT,		VarType::FLOAT	) );
			ops.push_back( ConstEvalOperation( div_n_by_n<float2>,			VarType::FLOAT2,	VarType::FLOAT2,	VarType::FLOAT2	) );
			ops.push_back( ConstEvalOperation( div_n_by_n<float3>,			VarType::FLOAT3,	VarType::FLOAT3,	VarType::FLOAT3	) );
			ops.push_back( ConstEvalOperation( div_n_by_n<float4>,			VarType::FLOAT4,	VarType::FLOAT4,	VarType::FLOAT4	) );
#if 0
			ops.push_back( ConstEvalOperation( div_n_by_1<float2,float>,	VarType::FLOAT2,	VarType::FLOAT2,	VarType::FLOAT	) );
			ops.push_back( ConstEvalOperation( div_n_by_1<float3,float>,	VarType::FLOAT3,	VarType::FLOAT3,	VarType::FLOAT	) );
			ops.push_back( ConstEvalOperation( div_n_by_1<float4,float>,	VarType::FLOAT4,	VarType::FLOAT4,	VarType::FLOAT	) );
#endif

			MD_ADD_ITEM("/");
		}

		// float3
		{
			Operations ops;
			ops.push_back( ConstEvalOperation( conv_4_to_3_f,		VarType::FLOAT3,		VarType::FLOAT4		) );

			MD_ADD_ITEM("(float3)");
		}

		// float3
		{
			Operations ops;
			ops.push_back( ConstEvalOperation( conv_4vec_to_3vec_f,	VarType::FLOAT3_VEC,	VarType::FLOAT4_VEC	) );

			MD_ADD_ITEM("(float3_vec)");
		}

		// max
		{
			Operations ops;

			ops.push_back( ConstEvalOperation( max_t<float>,	VarType::FLOAT,		VarType::FLOAT,		VarType::FLOAT	) );
			ops.push_back( ConstEvalOperation( max_t<float2>,	VarType::FLOAT2,	VarType::FLOAT2,	VarType::FLOAT2	) );
			ops.push_back( ConstEvalOperation( max_t<float3>,	VarType::FLOAT3,	VarType::FLOAT3,	VarType::FLOAT3	) );
			ops.push_back( ConstEvalOperation( max_t<float4>,	VarType::FLOAT4,	VarType::FLOAT4,	VarType::FLOAT4	) );

			ops.push_back( ConstEvalOperation( max_t<int>,		VarType::INT,		VarType::INT,		VarType::INT	) );
			ops.push_back( ConstEvalOperation( max_t<int2>,		VarType::INT2,		VarType::INT2,		VarType::INT2	) );
			ops.push_back( ConstEvalOperation( max_t<int3>,		VarType::INT3,		VarType::INT3,		VarType::INT3	) );
			ops.push_back( ConstEvalOperation( max_t<int4>,		VarType::INT4,		VarType::INT4,		VarType::INT4	) );

			ops.push_back( ConstEvalOperation( max_t<uint>,		VarType::UINT,		VarType::UINT,		VarType::UINT	) );
			ops.push_back( ConstEvalOperation( max_t<uint2>,	VarType::UINT2,		VarType::UINT2,		VarType::UINT2	) );
			ops.push_back( ConstEvalOperation( max_t<uint3>,	VarType::UINT3,		VarType::UINT3,		VarType::UINT3	) );
			ops.push_back( ConstEvalOperation( max_t<uint4>,	VarType::UINT4,		VarType::UINT4,		VarType::UINT4	) );

			MD_ADD_ITEM("max(");
		}

		// min
		{
			Operations ops;

			ops.push_back( ConstEvalOperation( min_t<float>,	VarType::FLOAT,		VarType::FLOAT,		VarType::FLOAT	) );
			ops.push_back( ConstEvalOperation( min_t<float2>,	VarType::FLOAT2,	VarType::FLOAT2,	VarType::FLOAT2	) );
			ops.push_back( ConstEvalOperation( min_t<float3>,	VarType::FLOAT3,	VarType::FLOAT3,	VarType::FLOAT3	) );
			ops.push_back( ConstEvalOperation( min_t<float4>,	VarType::FLOAT4,	VarType::FLOAT4,	VarType::FLOAT4	) );

			ops.push_back( ConstEvalOperation( min_t<int>,		VarType::INT,		VarType::INT,		VarType::INT	) );
			ops.push_back( ConstEvalOperation( min_t<int2>,		VarType::INT2,		VarType::INT2,		VarType::INT2	) );
			ops.push_back( ConstEvalOperation( min_t<int3>,		VarType::INT3,		VarType::INT3,		VarType::INT3	) );
			ops.push_back( ConstEvalOperation( min_t<int4>,		VarType::INT4,		VarType::INT4,		VarType::INT4	) );

			ops.push_back( ConstEvalOperation( min_t<uint>,		VarType::UINT,		VarType::UINT,		VarType::UINT	) );
			ops.push_back( ConstEvalOperation( min_t<uint2>,	VarType::UINT2,		VarType::UINT2,		VarType::UINT2	) );
			ops.push_back( ConstEvalOperation( min_t<uint3>,	VarType::UINT3,		VarType::UINT3,		VarType::UINT3	) );
			ops.push_back( ConstEvalOperation( min_t<uint4>,	VarType::UINT4,		VarType::UINT4,		VarType::UINT4	) );

			MD_ADD_ITEM("min(");
		}

#undef MD_ADD_ITEM


	}

	//------------------------------------------------------------------------

	EXP_IMP
	ConstEvalOperationGroupProvider::~ConstEvalOperationGroupProvider()
	{

	}


}