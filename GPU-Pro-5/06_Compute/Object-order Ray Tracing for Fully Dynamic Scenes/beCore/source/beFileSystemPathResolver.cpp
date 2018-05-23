/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beFileSystemPathResolver.h"

#include "beCore/beFileSystem.h"

#include <lean/io/filesystem.h>

namespace beCore
{

/// Constructor.
FileSystemPathResolver::FileSystemPathResolver(const utf8_ntri &location)
	: m_location( location.to<utf8_string>() )
{
}

// Constructor.
FileSystemPathResolver::FileSystemPathResolver(const FileSystemPathResolver &right)
	: m_location( right.m_location )
{
}

// Destructor.
FileSystemPathResolver::~FileSystemPathResolver()
{
}

// Resolves the given file name.
Exchange::utf8_string FileSystemPathResolver::Resolve(const utf8_ntri &file, bool bThrow) const
{
	Exchange::utf8_string filePath;

	if (lean::file_exists(file))
		filePath.assign(file.begin(), file.end());
	else
		filePath = beCore::FileSystem::Get().Search(m_location, file, bThrow);

	filePath = lean::absolute_path<Exchange::utf8_string>(filePath, lean::initial_directory());

	return filePath;
}

// Shortens the given path.
Exchange::utf8_string FileSystemPathResolver::Shorten(const utf8_ntri &path) const
{
	bool bMatch = false;
	Exchange::utf8_string result = beCore::FileSystem::Get().Shorten(m_location, path, &bMatch);

	if (!bMatch)
		result = lean::relative_path<Exchange::utf8_string>(path, lean::initial_directory(), true);

	return result;
}

/// Constructs and returns a clone of this path resolver.
FileSystemPathResolver* FileSystemPathResolver::clone() const
{
	return new FileSystemPathResolver(*this);
}
/// Destroys an include manager.
void FileSystemPathResolver::destroy() const
{
	delete this;
}

} // namespace