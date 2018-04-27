#ifndef COMMON_FORW_H_INCLUDED
#define COMMON_FORW_H_INCLUDED

namespace Mod
{
#include "DefDeclareClass.h"

	MOD_DECLARE_CLASS(XMLDoc)
	MOD_DECLARE_CLASS(XMLElem);
	MOD_DECLARE_CLASS(XMLAttrib);
	MOD_DECLARE_CLASS(TypedParam)

#include "UndefDeclareClass.h"	

	namespace VarType
	{
		enum Type;
	}

	struct Color;
}

#endif