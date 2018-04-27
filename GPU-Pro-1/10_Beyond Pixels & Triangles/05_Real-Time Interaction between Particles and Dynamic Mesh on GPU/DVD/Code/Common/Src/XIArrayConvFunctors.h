#ifndef COMMON_XIARRAYCONVFUNCTORS_H_INCLUDED
#define COMMON_XIARRAYCONVFUNCTORS_H_INCLUDED

#include "XIAttribute.h"

namespace Mod
{

	template <typename T>
	class XIAttribConv
	{
		// construction/ destruction
	public:
		XIAttribConv(const String& attName):mAttName(attName) 
		{}

		T operator()(const XMLElemPtr& elem) const
		{
			return XIAttribute<T>(elem, mAttName);
		}

		// data
	private:
		String mAttName;

	};

	//------------------------------------------------------------------------

	template <typename T>
	struct XIStringConvertibleElemConv
	{
		T operator()( const XMLElemPtr& elem ) const
		{
			return elem->GetName();
		}
	};

}

#endif