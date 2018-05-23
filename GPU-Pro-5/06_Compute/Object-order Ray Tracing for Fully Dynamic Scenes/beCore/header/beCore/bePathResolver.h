/******************************************************/
/* breeze Engine Core Module     (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_CORE_PATH_RESOLVER
#define BE_CORE_PATH_RESOLVER

#include "beCore.h"
#include "beExchangeContainers.h"
#include <lean/smart/cloneable.h>

namespace beCore
{

/// Path resolver interface.
class LEAN_INTERFACE PathResolver : public lean::cloneable
{
	LEAN_INTERFACE_BEHAVIOR(PathResolver)

public:
	/// Resolves the given file name.
	virtual Exchange::utf8_string Resolve(const utf8_ntri &file, bool bThrow = false) const = 0;
	/// Shortens the given path.
	virtual Exchange::utf8_string Shorten(const utf8_ntri &path) const = 0;
};

} // namespace

#endif