/*****************************************************/
/* lean Strings                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_STRINGS_STRINGS
#define LEAN_STRINGS_STRINGS

namespace lean
{
	/// Provides a generic character range type as well as conversion and streaming
	/// facilities to both simplify and generalize string passing and handling in
	/// your code.
	namespace strings { }
}

#include "char_traits.h"
#include "nullterminated.h"
#include "nullterminated_range.h"
#include "types.h"
#include "charstream.h"

#endif