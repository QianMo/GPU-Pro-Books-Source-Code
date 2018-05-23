/******************************************************/
/* breeze Engine Core Module     (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_CORE_FILE_CONTENT_PROVIDER
#define BE_CORE_FILE_CONTENT_PROVIDER

#include "beCore.h"
#include "beContentProvider.h"

namespace beCore
{

class FileContent;
template <class ContentType>
class DefaultContentProvider;

/// File content provider implementation.
template <>
class DefaultContentProvider<FileContent> : public ContentProvider
{
public:
	/// Gets the content identified by the given path.
	BE_CORE_API lean::com_ptr<Content, true> GetContent(const utf8_ntri &file);
	
	/// Gets a revision number for the content identified by the given path.
	BE_CORE_API uint8 GetRevision(const utf8_ntri &file) const;

	/// Constructs and returns a clone of this path resolver.
	BE_CORE_API DefaultContentProvider* clone() const;
	/// Destroys an include manager.
	BE_CORE_API void destroy() const;
};

/// File content provider.
typedef class DefaultContentProvider<FileContent> FileContentProvider;

} // namespace

#endif