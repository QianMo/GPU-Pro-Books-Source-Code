/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_FILESYSTEM
#define BE_CORE_FILESYSTEM

#include "beCore.h"
#include "beExchangeContainers.h"
#include <lean/tags/noncopyable.h>
#include <lean/pimpl/pimpl_ptr.h>
#include <lean/strings/types.h>
#include <lean/rapidxml/rapidxml.hpp>

namespace beCore
{

/// File system class that allows for the search of files in specific directories.
class FileSystem : public lean::noncopyable
{
private:
	class Impl;
	lean::pimpl_ptr<Impl> m_impl;

	/// Constructor.
	FileSystem();
	/// Destructor.
	~FileSystem();

public:
	/// Path string type.
	typedef Exchange::utf8_string path_type;
	/// Path list type.
	typedef Exchange::list_t<path_type>::t path_list;

	/// Gets the file system.
	BE_CORE_API static FileSystem& Get();

	/// Adds the given path to the given virtual location.
	BE_CORE_API void AddPath(const lean::utf8_ntri &location, const lean::utf8_ntri &path);
	/// Removes the given path from the given virtual location.
	BE_CORE_API void RemovePath(const lean::utf8_ntri &location, const lean::utf8_ntri &path);
	/// Removes the given virtual location.
	BE_CORE_API void RemoveLocation(const lean::utf8_ntri &location);
	/// Removes all given virtual locations.
	BE_CORE_API void Clear();

	/// Checks if the given virtual location exists.
	BE_CORE_API bool HasLocation(const lean::utf8_ntri &location) const;
	/// Gets the first path in the given location.
	BE_CORE_API Exchange::utf8_string GetPrimaryPath(const lean::utf8_ntri &location, bool bThrow = false) const;

	/// Searches for the given file or directory in the given virtual location.
	BE_CORE_API Exchange::utf8_string Search(const lean::utf8_ntri &location, const lean::utf8_ntri &file, bool bThrow = false) const;

	/// Shortens the given path, returning a path relative to the given location, if possible.
	BE_CORE_API Exchange::utf8_string Shorten(const lean::utf8_ntri &location, const lean::utf8_ntri &file, bool *pMatch = nullptr) const;

	/// Loads a configuration from the given file.
	BE_CORE_API void LoadConfiguration(const lean::utf8_ntri &file);
	/// Saves the current configuration to the given file.
	BE_CORE_API void SaveConfiguration(const lean::utf8_ntri &file) const;
	/// Loads a configuration from the given node.
	BE_CORE_API void LoadConfiguration(const rapidxml::xml_node<lean::utf8_t> &node);
	/// Saves the current configuration to the given node.
	BE_CORE_API void SaveConfiguration(rapidxml::xml_node<lean::utf8_t> &node) const;
};

} // namespace

#endif