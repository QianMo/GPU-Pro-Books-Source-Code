#ifndef PROVIDERS_CONSTEVALOPERATIONGROUP_H_INCLUDED
#define PROVIDERS_CONSTEVALOPERATIONGROUP_H_INCLUDED

#include "Forw.h"

#include "Common/Src/Named.h"

#include "ConstEvalNodeConfig.h"

namespace Mod
{
	struct ConstEvalOperation
	{
		typedef ConstEvalNodeConfig::Operation OperationFunc;
		typedef Types< VarType::Type > :: Vec VarTypes;

		ConstEvalOperation( OperationFunc a_func, VarType::Type a_returnType, VarType::Type a_type1 );
		ConstEvalOperation( OperationFunc a_func, VarType::Type a_returnType, VarType::Type a_type1, VarType::Type a_type2 );
		ConstEvalOperation( OperationFunc a_func, VarType::Type a_returnType, const VarTypes& a_types );

		OperationFunc	func;

		VarTypes		types;
		VarType::Type	returnType;
	};

	class ConstEvalOperationGroup : public Named
	{
		// types
	public:
		typedef Types< ConstEvalOperation > :: Vec	Operations;
		typedef Named								Base;
		typedef ConstEvalOperation::VarTypes		VarTypes;

		// construction/ destruction
	public:
		ConstEvalOperationGroup( const String& name, const Operations& ops );
		~ConstEvalOperationGroup();

		// manipulation/ access
	public:
		const ConstEvalOperation& GetByTypes( VarType::Type type1 ) const;
		const ConstEvalOperation& GetByTypes( VarType::Type type1, VarType::Type type2 ) const;
		const ConstEvalOperation& GetByTypes( const VarTypes& types ) const;

		// data
	private:
		Operations	mOperations;
	};

}

#endif