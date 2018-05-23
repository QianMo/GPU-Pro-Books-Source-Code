/******************************************************/
/* breeze Engine Core Module     (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_CORE_CONTENT_PROVIDER
#define BE_CORE_CONTENT_PROVIDER

#include "beCore.h"
#include <lean/smart/cloneable.h>
#include "beContent.h"
#include <lean/smart/com_ptr.h>

namespace beCore
{

/// Content provider interface.
class LEAN_INTERFACE ContentProvider : public lean::cloneable
{
	LEAN_INTERFACE_BEHAVIOR(ContentProvider)

public:
	/// Gets the content identified by the given path.
	virtual lean::com_ptr<Content, true> GetContent(const utf8_ntri &file) = 0;

	/// Gets a revision number for the content identified by the given path.
	virtual uint8 GetRevision(const utf8_ntri &file) const = 0;
};

} // namespace

#endif