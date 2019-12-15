/*
	(c) Sergey Ryazanov (http://home.onego.ru/~ryazanov)

	Fast delegate compatible with C++ Standard.
*/
#ifndef SRUTIL_DELEGATE_INCLUDED
#define SRUTIL_DELEGATE_INCLUDED

namespace srutil
{
#ifdef SRUTIL_DELEGATE_PREFERRED_SYNTAX
	template <typename TSignature> class delegate;
	template <typename TSignature> class delegate_invoker;
#endif
}

#ifdef _MSC_VER
#define SRUTIL_DELEGATE_CALLTYPE __fastcall
#else
#define SRUTIL_DELEGATE_CALLTYPE
#endif

#include "detail/delegate_list.hpp"

#undef SRUTIL_DELEGATE_CALLTYPE

#endif//SRUTIL_DELEGATE_INCLUDED
