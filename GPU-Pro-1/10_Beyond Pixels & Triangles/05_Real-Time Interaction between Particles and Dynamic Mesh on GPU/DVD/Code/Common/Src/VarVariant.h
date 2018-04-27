#ifndef COMMON_VARVARIANT_H_INLCLUDED
#define COMMON_VARVARIANT_H_INLCLUDED

#include "Math/Src/Types.h"

#include "Common/Src/VarType.h"

namespace Mod
{
	namespace VarVariantNS
	{

		template <size_t A, size_t B, bool G>
		struct max_impl;

		template <size_t A, size_t B>	struct max_impl< A, B, true >	{		static const size_t Result = A;		};
		template <size_t A, size_t B>	struct max_impl< A, B, false >	{		static const size_t Result = B;		};

		template <size_t A, size_t B>
		struct max
		{
			static const size_t Result = max_impl< A, B, (A > B) > :: Result;
		};
	}

	class VarVariant
	{
		// construction/ destruction
	public:
		explicit VarVariant( VarType::Type type );
		VarVariant( const VarVariant& cpy );
		~VarVariant();

		// manipulation/ access
	public:

		VarVariant& operator = ( const VarVariant& var );


		template <typename T>
		void Set( const T& val );

		template <typename T>
		const T& Get() const;

		template <typename T>
		T* GetPtr();

		VarType::Type GetType() const;

		// data
	private:
		char mData[ VarVariantNS::max< sizeof( Types< Math::float4x4 > :: Vec ), sizeof(Math::float4x4) > :: Result ];

		VarType::Type	mType;

	};

	//------------------------------------------------------------------------

	template <typename T>
	void
	VarVariant::Set( const T& val )
	{
		MD_FERROR_ON_FALSE( VarType::TypeToEnum<T>::Result == mType );
		*reinterpret_cast<T*>( mData ) = val;
	}


	//------------------------------------------------------------------------

	template <typename T>
	const T&
	VarVariant::Get() const
	{		
		MD_FERROR_ON_FALSE( VarType::TypeToEnum<T>::Result == mType );
		return *reinterpret_cast<const T*>( mData );
	}

	//------------------------------------------------------------------------

	template <typename T>
	T*
	VarVariant::GetPtr()
	{
		MD_FERROR_ON_FALSE( VarType::TypeToEnum<T>::Result == mType );
		return reinterpret_cast<T*>( mData );		
	}


}

#endif