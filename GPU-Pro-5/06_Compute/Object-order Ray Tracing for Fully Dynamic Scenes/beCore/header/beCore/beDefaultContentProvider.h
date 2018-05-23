/******************************************************/
/* breeze Engine Core Module     (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_CORE_DEFAULT_CONTENT_PROVIDER
#define BE_CORE_DEFAULT_CONTENT_PROVIDER

#include "beCore.h"
#include "beContentProvider.h"

namespace beCore
{

/// Default content provider implementation.
template <class ContentType>
class DefaultContentProvider : public ContentProvider
{
public:
	/// Gets the content identified by the given path.
	lean::com_ptr<Content, true> GetContent(const utf8_ntri &file)
	{
		return lean::bind_com( new ContentType(file) );
	}

	/// Gets a revision number for the content identified by the given path.
	uint8 GetRevision(const utf8_ntri &file) const
	{
		return 0;
	}

	/// Constructs and returns a clone of this path resolver.
	DefaultContentProvider* clone() const
	{
		return new DefaultContentProvider(*this);
	}
	/// Destroys an include manager.
	void destroy() const
	{
		delete this;
	}
};

} // namespace

#endif