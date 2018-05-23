/******************************************************/
/* breeze Engine Core Module     (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_CORE_FILE_CONTENT
#define BE_CORE_FILE_CONTENT

#include "beCore.h"
#include "beContent.h"
#include <lean/tags/noncopyable.h>
#include <lean/io/mapped_file.h>

namespace beCore
{

/// File content class.
class FileContent : public lean::noncopyable, public Content
{
private:
	lean::rmapped_file m_file;

public:
	/// Constructor.
	FileContent(const utf8_ntri &file)
		: m_file(file)
	{
		m_memory = m_file.data();
		m_size = m_file.size();
	}
};

} // namespace

#endif