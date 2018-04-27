#include "Precompiled.h"
#include "VarVariant.h"

namespace Mod
{

	namespace
	{
		class TypeHelper
		{
			// construction/ destruction
		private:
			TypeHelper();
			~TypeHelper();

			// manipulation/ access
		public:
			static TypeHelper& Single();
			void Create( void* ptr, VarType::Type type ) const;
			void Destroy( void* ptr, VarType::Type type ) const;
			void Copy( void* dest, const void* src, VarType::Type type ) const;

			// data
		private:
			void (*mCreateFuncArr[VarType::NUM_TYPES])( void* ptr );
			void (*mDestroyFuncArr[VarType::NUM_TYPES])( void* ptr );
			void (*mCopyFuncArr[VarType::NUM_TYPES])( void* dest, const void* src );
		};
	}

	VarVariant::VarVariant( VarType::Type type ) :
	mType( type )
	{
		TypeHelper::Single().Create( mData, type );
	}

	//------------------------------------------------------------------------

	VarVariant::VarVariant( const VarVariant& cpy ) :
	mType( cpy.mType )
	{
		TypeHelper::Single().Copy( mData, cpy.mData, cpy.mType );
	}

	//------------------------------------------------------------------------

	VarVariant::~VarVariant()
	{
		TypeHelper::Single().Destroy( mData, mType );
	}

	//------------------------------------------------------------------------

	VarVariant& VarVariant::operator = ( const VarVariant& rhs )
	{
		MD_FERROR_ON_FALSE( mType == rhs.mType );
		TypeHelper::Single().Copy( mData, rhs.mData, rhs.mType );
		return *this;
	}

	//------------------------------------------------------------------------

	VarType::Type
	VarVariant::GetType() const
	{
		return mType;
	}

	//------------------------------------------------------------------------

	namespace
	{
		template < typename T >
		void CreateImpl( void *val )
		{
			new ( val ) T;
		}

		template < typename T >
		void DestroyImpl( void *val )
		{
			val; // MSVC ist wirklich verbugt ;]
			static_cast< T* > ( val ) -> ~T();
		}

		template < typename T >
		void CopyImpl( void * dest, const void* src )
		{
			*static_cast< T* > (dest) = *static_cast< const T* >( src );
		}


		TypeHelper::TypeHelper()
		{
			using namespace VarType;

			for( size_t i = 0, e = sizeof(mCreateFuncArr)/sizeof(mCreateFuncArr[0]); i < e; i++ )
			{
				mCreateFuncArr[ i ]		= NULL;
				mDestroyFuncArr[ i ]	= NULL;
				mCopyFuncArr[ i ]		= NULL;
			}
#define MD_INSERT_TYPE_FUNCS(type)		{ mCreateFuncArr[type] = CreateImpl< VarType::EnumToType< type > :: Result > ; mDestroyFuncArr[type] = DestroyImpl< VarType::EnumToType< type > :: Result > ; mCopyFuncArr[type] = CopyImpl< VarType::EnumToType< type > :: Result >; }
#define MD_INSERT_TYPE_ROW_FUNCS(type)	MD_INSERT_TYPE_FUNCS(type)			MD_INSERT_TYPE_FUNCS(type##2)		MD_INSERT_TYPE_FUNCS(type##3)		MD_INSERT_TYPE_FUNCS(type##4)		MD_INSERT_TYPE_FUNCS(type##2x2)		MD_INSERT_TYPE_FUNCS(type##3x3)		MD_INSERT_TYPE_FUNCS(type##4x4)		MD_INSERT_TYPE_FUNCS(type##3x4)\
										MD_INSERT_TYPE_FUNCS(type##_VEC)	MD_INSERT_TYPE_FUNCS(type##2_VEC)	MD_INSERT_TYPE_FUNCS(type##3_VEC)	MD_INSERT_TYPE_FUNCS(type##4_VEC)	MD_INSERT_TYPE_FUNCS(type##2x2_VEC)	MD_INSERT_TYPE_FUNCS(type##3x3_VEC)	MD_INSERT_TYPE_FUNCS(type##4x4_VEC)	MD_INSERT_TYPE_FUNCS(type##3x4_VEC)

			MD_INSERT_TYPE_ROW_FUNCS(FLOAT)
			MD_INSERT_TYPE_ROW_FUNCS(UINT)
			MD_INSERT_TYPE_ROW_FUNCS(INT)

#undef MD_INSERT_TYPE_ROW_FUNCS
#undef MD_INSERT_TYPE_FUNCS
		}

		//------------------------------------------------------------------------

		TypeHelper::~TypeHelper()
		{

		}

		//------------------------------------------------------------------------

		TypeHelper&
		TypeHelper::Single()
		{
			static TypeHelper single;
			return single;
		}

		//------------------------------------------------------------------------

		void
		TypeHelper::Create( void* ptr, VarType::Type type ) const
		{
			MD_FERROR_ON_FALSE( (UINT32)type < VarType::NUM_TYPES );
			MD_FERROR_ON_FALSE( mCreateFuncArr[ type ] );
			mCreateFuncArr[ type ]( ptr );
		}

		//------------------------------------------------------------------------

		void
		TypeHelper::Destroy( void* ptr, VarType::Type type ) const
		{
			MD_FERROR_ON_FALSE( (UINT32)type < VarType::NUM_TYPES );
			MD_FERROR_ON_FALSE( mDestroyFuncArr[ type ] );
			mDestroyFuncArr[ type ]( ptr );
		}

		//------------------------------------------------------------------------

		void
		TypeHelper::Copy( void* dest, const void* src, VarType::Type type ) const
		{
			MD_FERROR_ON_FALSE( (UINT32)type < VarType::NUM_TYPES );
			mCopyFuncArr[ type ]( dest, src );
		}

	}

}