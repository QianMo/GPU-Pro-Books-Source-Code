#ifndef PROVIDERS_CONSTEVALNODE_H_INCLUDED
#define PROVIDERS_CONSTEVALNODE_H_INCLUDED

#include "Common/Src/VarVariant.h"

#include "ConstEvalNodeConfig.h"

#include "ExportDefs.h"

#define MD_NAMESPACE ConstEvalNodeNS
#include "ConfigurableImpl.h"


namespace Mod
{
	class ConstEvalNode :	public ConstEvalNodeNS::ConfigurableImpl<ConstEvalNodeConfig>
	{
		// types
	public:
		typedef ConstEvalNodeConfig::Parents	Parents;
		typedef ConstEvalNodeConfig::Children	Children;
		typedef ConstEvalNodeConfig::Operation	Operation;

		// construction/ destruction
	public:
		explicit ConstEvalNode( const ConstEvalNodeConfig& cfg);
		~ConstEvalNode();

		// manipulation/ access
	public:
		EXP_IMP const String&		GetName() const;
		EXP_IMP const VarVariant&	GetVar();
		EXP_IMP VarType::Type		GetType() const;
		
		void				LinkParent( const ConstEvalNodePtr& parent );

		// to be called by ConstEvalNodeProvider only
		static void			Set( ConstEvalNodePtr rhs, ConstEvalNodePtr lhs );

		template <typename T>
		T GetVal();

		template <typename T>
		T&
		GetModifiableVal();

		template <typename T>
		void SetVal( const T& val );

		// helpers
	private:
		EXP_IMP	void Update();
		EXP_IMP void MarkDirty();

		void UpdateParent( ConstEvalNodePtr oldParent, ConstEvalNodePtr newParent );
		// data
	private:
		VarVariant	mVariant;
		Operation	mOperation;

		Parents		mParents;
		Children	mChildren;

		bool		mDirty;
	};

	//------------------------------------------------------------------------

	template <typename T>
	T
	ConstEvalNode::GetVal()
	{
		Update();
		return mVariant.Get<T>();
	}

	//------------------------------------------------------------------------

	template <typename T>
	T&
	ConstEvalNode::GetModifiableVal()
	{
		MarkDirty();
		return *mVariant.GetPtr<T>();
	}

	//------------------------------------------------------------------------

	template <typename T>
	void
	ConstEvalNode::SetVal( const T& val )
	{
		MarkDirty();
		mVariant.Set( val );
	}


}

#endif