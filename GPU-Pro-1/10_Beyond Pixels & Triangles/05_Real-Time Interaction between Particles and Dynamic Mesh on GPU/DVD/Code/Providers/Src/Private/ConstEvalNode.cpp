#include "Precompiled.h"

#include "Math/Src/Operations.h"

#include "ConstEvalNode.h"

#define MD_NAMESPACE ConstEvalNodeNS
#include "ConfigurableImpl.cpp.h"


namespace Mod
{

	template class ConstEvalNodeNS::ConfigurableImpl<ConstEvalNodeConfig>;

	//------------------------------------------------------------------------

	ConstEvalNode::ConstEvalNode( const ConstEvalNodeConfig& cfg) : 
	Parent( cfg ),
	mDirty( true ),
	mVariant( cfg.type ),
	mChildren( cfg.children ),
	mOperation( cfg.operation )
	{

	}

	//------------------------------------------------------------------------

	ConstEvalNode::~ConstEvalNode()
	{

	}

	//------------------------------------------------------------------------

	EXP_IMP
	const String&
	ConstEvalNode::GetName() const
	{
		return GetConfig().name;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	const VarVariant&
	ConstEvalNode::GetVar()
	{
		Update();
		return mVariant;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	VarType::Type
	ConstEvalNode::GetType() const
	{
		return mVariant.GetType();
	}

	//------------------------------------------------------------------------

	void
	ConstEvalNode::LinkParent( const ConstEvalNodePtr& parent )
	{
		if( std::find( mParents.begin(), mParents.end(), parent ) == mParents.end() )
		{
			mParents.push_back( parent );
		}
	}

	//------------------------------------------------------------------------

	void
	ConstEvalNode::Set( ConstEvalNodePtr rhs, ConstEvalNodePtr lhs )
	{
		MD_FERROR_ON_FALSE( lhs )

		rhs->mChildren = lhs->mChildren;
		rhs->mOperation = lhs->mOperation;

		MD_FERROR_ON_FALSE( lhs->mVariant.GetType() == rhs->mVariant.GetType() );
		
		rhs->mVariant = lhs->mVariant;

		rhs->MarkDirty();

		for( size_t i = 0, e = rhs->mChildren.size(); i < e; i ++ )
		{
			rhs->mChildren[i]->UpdateParent( lhs, rhs );
		}

	}

	//------------------------------------------------------------------------

	EXP_IMP
	void
	ConstEvalNode::Update()
	{
		if( mDirty )
		{
			if( mOperation )
			{
				mOperation( mVariant, mChildren );
			}

			mDirty = false;
		}
	}

	//------------------------------------------------------------------------

	EXP_IMP
	void
	ConstEvalNode::MarkDirty()
	{
		mDirty = true;

		for( size_t i = 0, e = mParents.size(); i < e; i ++ )
			mParents[i]->MarkDirty();
	}

	//------------------------------------------------------------------------

	void
	ConstEvalNode::UpdateParent( ConstEvalNodePtr oldParent, ConstEvalNodePtr newParent )
	{
		for( size_t i = 0, e = mParents.size(); i < e; i ++ )
		{
			if( mParents[i] == oldParent )
			{
				mParents[i] = newParent;
				break;
			}
		}
	}

}