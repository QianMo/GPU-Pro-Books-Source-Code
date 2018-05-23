/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beDefaultPathResolver.h"

#include <lean/io/filesystem.h>

namespace beCore
{

// Resolves the given file name.
Exchange::utf8_string DefaultPathResolver::Resolve(const utf8_ntri &file, bool bThrow) const
{
	return lean::absolute_path<Exchange::utf8_string>(file, lean::initial_directory());
}

// Shortens the given path.
Exchange::utf8_string DefaultPathResolver::Shorten(const utf8_ntri &path) const
{
	return lean::relative_path<Exchange::utf8_string>(path, lean::initial_directory(), true);
}

/// Constructs and returns a clone of this path resolver.
DefaultPathResolver* DefaultPathResolver::clone() const
{
	return new DefaultPathResolver(*this);
}
/// Destroys an include manager.
void DefaultPathResolver::destroy() const
{
	delete this;
}

} // namespace