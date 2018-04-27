#include "Precompiled.h"
#include "Named.h"

namespace Mod
{
	template <typename S>
	BasicNamed<S>::BasicNamed(const StringType& name):
	mName(name)
	{

	}

	//------------------------------------------------------------------------

	template <typename S>
	BasicNamed<S>::~BasicNamed()
	{

	}

	//------------------------------------------------------------------------

	template <typename S>
	const typename BasicNamed<S>::StringType&
	BasicNamed<S>::GetName() const
	{
		return mName;
	}

	//------------------------------------------------------------------------

	template <typename S>
	const typename BasicNamed<S>::StringType::value_type*
	BasicNamed<S>::GetCName() const
	{
		return mName.c_str();
	}

	//------------------------------------------------------------------------

	template <typename S>
	void
	BasicNamed<S>::Rename(const StringType& name)
	{
		mName = name;
	}

	//------------------------------------------------------------------------

	template Named;
	template AnsiNamed;

}

