/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beFileContentProvider.h"
#include "beCore/beFileContent.h"

#include <lean/io/filesystem.h>

namespace beCore
{

// Gets the content identified by the given path.
lean::com_ptr<Content, true> FileContentProvider::GetContent(const utf8_ntri &file)
{
	return lean::bind_com( new FileContent(file) );
}

// Gets a revision number for the content identified by the given path.
uint8 FileContentProvider::GetRevision(const utf8_ntri &file) const
{
	return lean::file_revision(file);
}

// Constructs and returns a clone of this path resolver.
FileContentProvider* FileContentProvider::clone() const
{
	return new FileContentProvider(*this);
}
// Destroys an include manager.
void FileContentProvider::destroy() const
{
	delete this;
}

} // namespace