#ifndef COMMON_TYPED_PARAM_H_INCLUDED
#define COMMON_TYPED_PARAM_H_INCLUDED

#include "Forw.h"
#include "VarType.h"

namespace Mod
{
	class TypedParam : AntiValue
	{
		// construction/ destruction
	public:
		explicit TypedParam( VarType::Type type );
		virtual ~TypedParam();

		// manipulation/ access
	public:
		void SetVal( const String& val );

		template< typename T >
		void SetVal( const T& val );

		template< typename T >
		const T& GetVal() const;

		VarType::Type GetType() const;

		// polymorphism
	private:
		virtual void		SetValImpl( const void* val ) = 0;
		virtual const void*	GetValImpl() const = 0;

		// data
	private:
		VarType::Type mType;
	};

	//------------------------------------------------------------------------

	template< typename T >
	void
	TypedParam::SetVal( const T& val )
	{
		MD_FERROR_ON_FALSE( VarType :: TypeToEnum<T> :: Result == mType );
		SetValImpl( &val );
	}

	//------------------------------------------------------------------------

	template< typename T >
	const T&
	TypedParam::GetVal() const
	{
		MD_FERROR_ON_FALSE( VarType :: TypeToEnum<T> :: Result == mType );
		return *static_cast<const T*>( GetValImpl() );
	}

	//------------------------------------------------------------------------

	TypedParamPtr CreateTypedParam( VarType::Type type );
	TypedParamPtr CreateTypedParam( const String& type );

}

#endif