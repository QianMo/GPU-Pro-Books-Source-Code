#include "Precompiled.h"

#include "ConstEvalOperationGroup.h"

namespace Mod
{

	ConstEvalOperation::ConstEvalOperation( OperationFunc a_func, VarType::Type a_returnType, VarType::Type a_type1 ) :
	func( a_func ),
	returnType( a_returnType )
	{
		types.push_back( a_type1 );
	}

	//------------------------------------------------------------------------

	ConstEvalOperation::ConstEvalOperation( OperationFunc a_func, VarType::Type a_returnType, VarType::Type a_type1, VarType::Type a_type2 ) :
	func( a_func ),
	returnType( a_returnType )
	{
		types.push_back( a_type1 );
		types.push_back( a_type2 );
	}

	//------------------------------------------------------------------------

	ConstEvalOperation::ConstEvalOperation( OperationFunc a_func, VarType::Type a_returnType, const VarTypes& a_types ) :
	types( a_types ),
	func( a_func ),
	returnType( a_returnType )
	{

	}

	//------------------------------------------------------------------------

	ConstEvalOperationGroup::ConstEvalOperationGroup( const String& name, const Operations& ops ) :
	Named( name ),
	mOperations( ops )
	{

	}

	//------------------------------------------------------------------------

	ConstEvalOperationGroup::~ConstEvalOperationGroup()
	{

	}

	//------------------------------------------------------------------------

	const ConstEvalOperation&
	ConstEvalOperationGroup::GetByTypes( VarType::Type type1 ) const
	{
		for( size_t i = 0, e = mOperations.size(); i < e; i ++ )
		{
			const ConstEvalOperation& op = mOperations[ i ];
			if( op.types.size() == 1 && op.types[0] == type1  )
				return op;
		}

		MD_FERROR( L"Couldnt find a function accepting required types" );

		return *(ConstEvalOperation*)0;
	}

	//------------------------------------------------------------------------

	const ConstEvalOperation&
	ConstEvalOperationGroup::GetByTypes( VarType::Type type1, VarType::Type type2 ) const
	{
		for( size_t i = 0, e = mOperations.size(); i < e; i ++ )
		{
			const ConstEvalOperation& op = mOperations[ i ];
			if( op.types.size() == 2 && op.types[0] == type1 && op.types[1] == type2 )
				return op;
		}

		MD_FERROR( L"Couldnt find a function accepting required types" );

		return *(ConstEvalOperation*)0;
	}

	//------------------------------------------------------------------------

	const ConstEvalOperation&
	ConstEvalOperationGroup::GetByTypes( const VarTypes& types ) const
	{
		for( size_t i = 0, e = mOperations.size(); i < e; i ++ )
		{
			const ConstEvalOperation& op = mOperations[ i ];

			size_t n = op.types.size();

			if( n == types.size() )
			{
				size_t i;
				for( i = 0; i < n; i ++  )
					if( op.types[i] != types[i] )
						break;

				if( i == n )
					return op;
			}
		}

		MD_FERROR( L"Couldnt find a function accepting required types" );

		return *(ConstEvalOperation*)0;
	}

}