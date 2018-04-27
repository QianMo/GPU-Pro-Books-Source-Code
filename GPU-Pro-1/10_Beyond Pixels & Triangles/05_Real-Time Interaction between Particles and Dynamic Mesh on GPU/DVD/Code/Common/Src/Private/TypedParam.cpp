#include "Precompiled.h"

#include "VarTypeParser.h"

#include "StringParsing.h"

#include "TypedParam.h"

namespace Mod
{
	namespace
	{
		void SetParmValue( TypedParam& parm, const String& val );
	}


	/*explicit*/
	TypedParam::TypedParam( VarType::Type type ) :
	mType( type )
	{

	}

	//------------------------------------------------------------------------

	/*virtual*/
	TypedParam::~TypedParam()
	{

	}

	//------------------------------------------------------------------------

	void
	TypedParam::SetVal( const String& val )
	{
		SetParmValue( *this, val );
	}

	//------------------------------------------------------------------------

	VarType::Type
	TypedParam::GetType() const
	{
		return mType;
	}

	//------------------------------------------------------------------------

	template <typename T>
	class TypedParamImpl : public TypedParam
	{
		// typedef
	public:
		typedef TypedParam Base;

		// construction/ destruction
	public:
		TypedParamImpl();

		// polymorphism
	private:
		virtual void		SetValImpl( const void* val ) OVERRIDE;
		virtual const void*	GetValImpl() const OVERRIDE;

		// data
	private:
		T mData;
	};

	//------------------------------------------------------------------------
	
	template < typename T >
	TypedParamImpl<T>::TypedParamImpl() :
	Base( VarType::TypeToEnum<T> :: Result )
	{

	}

	//------------------------------------------------------------------------
	/*virtual*/

	template <typename T>
	void
	TypedParamImpl<T>::SetValImpl( const void* val ) /*OVERRIDE*/
	{
		mData = *static_cast< const T* >( val );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	template <typename T>
	const void*
	TypedParamImpl<T>::GetValImpl() const /*OVERRIDE*/
	{
		return &mData;
	}

	//------------------------------------------------------------------------

	namespace
	{
		// avoid global vars ( that's now from experience, not books ;] )
		class ImplHelper
		{
			// construction/ destruction
		private:
			ImplHelper();			

			// manipulation/ access
		public:
			static ImplHelper& Single();
			TypedParamPtr	Create( VarType::Type type );
			void			SetValue( TypedParam& parm, const String& val );

			// data
		private:
			TypedParamPtr (*mCreateTypedParamImplArr[VarType::NUM_TYPES])();
			void (*mSetValueArr[VarType::NUM_TYPES])( TypedParam&, const String& );
		};
	}

	TypedParamPtr CreateTypedParam( VarType::Type type )
	{
		return ImplHelper::Single().Create( type );
	}

	//------------------------------------------------------------------------

	TypedParamPtr CreateTypedParam( const String& type )
	{
		return CreateTypedParam( VarTypeParser::Single().GetItem( type ) );
	}

	//------------------------------------------------------------------------

	namespace
	{
		template <typename T>
		TypedParamPtr CreateTypedParamImpl()
		{
			return TypedParamPtr( new TypedParamImpl<T>() );
		}

		//------------------------------------------------------------------------

		template <typename T>
		void SetParmValueImpl( TypedParam& parm, const String& val )
		{
			parm.SetVal( FromString<T>( val ) );
		}

		void InvalidStringSetParmValueImpl( TypedParam& , const String&  )
		{
			MD_FERROR( L"Unimplelemted!" );
		}

		//------------------------------------------------------------------------

		ImplHelper::ImplHelper()
		{
			for( int i = 0; i < VarType::NUM_TYPES; i ++ )
			{
				mCreateTypedParamImplArr[i]	= NULL;
				mSetValueArr[i]				= NULL;
			}

#define MD_INSERT_TYPE(e) MD_STATIC_ASSERT(e<VarType::NUM_TYPES); mCreateTypedParamImplArr[e] = CreateTypedParamImpl< VarType::EnumToType<e>::Result >; mSetValueArr[e] = SetParmValueImpl< VarType::EnumToType<e>::Result >;
#define MD_INSERT_TYPE_ROW(name)																			\
			MD_INSERT_TYPE(name) MD_INSERT_TYPE(name##2) MD_INSERT_TYPE(name##3) MD_INSERT_TYPE(name##4)	\
			MD_INSERT_TYPE(name##2x2) MD_INSERT_TYPE(name##3x3) MD_INSERT_TYPE(name##4x4)

		MD_INSERT_TYPE_ROW(VarType::FLOAT)
		MD_INSERT_TYPE_ROW(VarType::INT)
		MD_INSERT_TYPE_ROW(VarType::UINT)

#undef MD_INSERT_TYPE

		}

		//------------------------------------------------------------------------
		ImplHelper&
		ImplHelper::Single()
		{
			static ImplHelper creator;
			return creator;
		}

		//------------------------------------------------------------------------

		TypedParamPtr
		ImplHelper::Create( VarType::Type type )
		{
			MD_FERROR_ON_TRUE( type == VarType::UNKNOWN );

			return mCreateTypedParamImplArr[ type ]();
		}

		//------------------------------------------------------------------------

		void
		ImplHelper::SetValue( TypedParam& parm, const String& val )
		{
			return mSetValueArr[ parm.GetType() ] ( parm, val );
		}

		//------------------------------------------------------------------------

		void SetParmValue( TypedParam& parm, const String& val )
		{
			return ImplHelper::Single().SetValue( parm, val );
		}
	}



}