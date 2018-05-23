/******************************************************/
/* breeze Engine Core Module     (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_CORE_FILESYSTEM_PATH_RESOLVER
#define BE_CORE_FILESYSTEM_PATH_RESOLVER

#include "beCore.h"
#include "bePathResolver.h"
#include <lean/tags/noncopyable.h>

namespace beCore
{

/// File system path resolver.
class FileSystemPathResolver : public lean::nonassignable, public PathResolver
{
private:
	utf8_string m_location;

public:
	/// Constructor.
	BE_CORE_API FileSystemPathResolver(const utf8_ntri &location);
	/// Constructor.
	BE_CORE_API FileSystemPathResolver(const FileSystemPathResolver &right);
	/// Destructor.
	BE_CORE_API ~FileSystemPathResolver();

	// Resolves the given file name.
	BE_CORE_API Exchange::utf8_string Resolve(const utf8_ntri &file, bool bThrow = false) const;
	/// Shortens the given path.
	BE_CORE_API Exchange::utf8_string Shorten(const utf8_ntri &path) const;

	/// Constructs and returns a clone of this path resolver.
	BE_CORE_API FileSystemPathResolver* clone() const;
	/// Destroys an include manager.
	BE_CORE_API void destroy() const;
};

} // namespace

#endif