/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beFileSystem.h"

#include <lean/io/filesystem.h>
#include <lean/xml/xml_file.h>
#include <lean/xml/utility.h>
#include <unordered_map>
#include <lean/logging/errors.h>

/// Implementation of the file system class internals.
class beCore::FileSystem::Impl
{
private:
	typedef std::unordered_map<lean::utf8_string, path_list> location_map;
	location_map m_locations;

public:
	/// Constructor.
	Impl();
	
	/// Adds the given path to the given virtual location.
	void AddPath(const lean::utf8_ntri &location, const lean::utf8_ntri &path);
	/// Removes the given path from the given virtual location.
	void RemovePath(const lean::utf8_ntri &location, const lean::utf8_ntri &path);
	/// Removes the given virtual location.
	void RemoveLocation(const lean::utf8_ntri &location);
	/// Removes all given virtual locations.
	void Clear();

	/// Checks if the given virtual location exists.
	bool HasLocation(const lean::utf8_ntri &location) const;
	/// Gets the first path in the given location.
	Exchange::utf8_string GetPrimaryPath(const lean::utf8_ntri &location, bool bThrow) const;

	/// Searches for the given file or directory in the given virtual location.
	Exchange::utf8_string Search(const lean::utf8_ntri &location, const lean::utf8_ntri &file, bool bThrow) const;

	/// Shortens the given path, returning a path relative to the given location, if possible.
	Exchange::utf8_string Shorten(const lean::utf8_ntri &location, const lean::utf8_ntri &file, bool *pMatch) const;

	/// Loads a configuration from the given node.
	void LoadConfiguration(const rapidxml::xml_node<lean::utf8_t> &node);
	/// Saves the current configuration to the given node.
	void SaveConfiguration(rapidxml::xml_node<lean::utf8_t> &node) const;
};

// Constructor.
beCore::FileSystem::FileSystem()
	: m_impl(new Impl())
{
}

// Destructor.
beCore::FileSystem::~FileSystem()
{
}

// Gets the file system.
beCore::FileSystem& beCore::FileSystem::Get()
{
	static FileSystem instance;
	return instance;
}

// Adds the given path to the given virtual location.
void beCore::FileSystem::AddPath(const lean::utf8_ntri &location, const lean::utf8_ntri &path)
{
	m_impl->AddPath(location, path);
}
// Removes the given path from the given virtual location.
void beCore::FileSystem::RemovePath(const lean::utf8_ntri &location, const lean::utf8_ntri &path)
{
	m_impl->RemovePath(location, path);
}
// Removes the given virtual location.
void beCore::FileSystem::RemoveLocation(const lean::utf8_ntri &location)
{
	m_impl->RemoveLocation(location);
}
// Removes all given virtual locations.
void beCore::FileSystem::Clear()
{
	m_impl->Clear();
}

// Checks if the given virtual location exists.
bool beCore::FileSystem::HasLocation(const lean::utf8_ntri &location) const
{
	return m_impl->HasLocation(location);
}

// Gets the first path in the given location.
beCore::Exchange::utf8_string beCore::FileSystem::GetPrimaryPath(const lean::utf8_ntri &location, bool bThrow) const
{
	return m_impl->GetPrimaryPath(location, bThrow);
}

// Searches for the given file or directory in the given virtual location.
beCore::Exchange::utf8_string beCore::FileSystem::Search(const lean::utf8_ntri &location, const lean::utf8_ntri &file, bool bThrow) const
{
	return m_impl->Search(location, file, bThrow);
}

// Shortens the given path, returning a path relative to the given location, if possible.
beCore::Exchange::utf8_string beCore::FileSystem::Shorten(const lean::utf8_ntri &location, const lean::utf8_ntri &file, bool *pMatch) const
{
	return m_impl->Shorten(location, file, pMatch);
}

// Loads a configuration from the given file.
void beCore::FileSystem::LoadConfiguration(const lean::utf8_ntri &file)
{
	lean::xml_file<lean::utf8_t> xml(file);
	rapidxml::xml_node<lean::utf8_t> *root = xml.document().first_node("filesystem");

	if (root)
		LoadConfiguration(*root);
}

// Saves the current configuration to the given file.
void beCore::FileSystem::SaveConfiguration(const lean::utf8_ntri &file) const
{
	lean::xml_file<lean::utf8_t> xml;
	rapidxml::xml_node<lean::utf8_t> *root = lean::allocate_node<utf8_t>(xml.document(), "filesystem");

	// ORDER: Append FIRST, otherwise parent document == nullptr
	xml.document().append_node(root);
	SaveConfiguration(*root);

	xml.save(file);
}

// Loads a configuration from the given node.
void beCore::FileSystem::LoadConfiguration(const rapidxml::xml_node<lean::utf8_t> &node)
{
	m_impl->LoadConfiguration(node);
}

// Saves the current configuration to the given node.
void beCore::FileSystem::SaveConfiguration(rapidxml::xml_node<lean::utf8_t> &node) const
{
	m_impl->SaveConfiguration(node);
}

// Constructor.
LEAN_INLINE beCore::FileSystem::Impl::Impl()
{
}

// Adds the given path to the given virtual location.
LEAN_INLINE void beCore::FileSystem::Impl::AddPath(const lean::utf8_ntri &location, const lean::utf8_ntri &path)
{
	m_locations[location.to<lean::utf8_string>()].push_back(path.to<path_type>());
}
// Removes the given path from the given virtual location.
LEAN_INLINE void beCore::FileSystem::Impl::RemovePath(const lean::utf8_ntri &location, const lean::utf8_ntri &path)
{
	m_locations[location.to<lean::utf8_string>()].remove(path.to<path_type>());
}
// Removes the given virtual location.
LEAN_INLINE void beCore::FileSystem::Impl::RemoveLocation(const lean::utf8_ntri &location)
{
	m_locations.erase(location.to<lean::utf8_string>());
}
// Removes all given virtual locations.
LEAN_INLINE void beCore::FileSystem::Impl::Clear()
{
	m_locations.clear();
}

// Checks if the given virtual location exists.
bool beCore::FileSystem::Impl::HasLocation(const lean::utf8_ntri &location) const
{
	location_map::const_iterator itLocation = m_locations.find(location.to<lean::utf8_string>());

	return (itLocation != m_locations.end() && !itLocation->second.empty());
}

// Gets the first path in the given location.
LEAN_INLINE beCore::Exchange::utf8_string beCore::FileSystem::Impl::GetPrimaryPath(const lean::utf8_ntri &location, bool bThrow) const
{
	Exchange::utf8_string result;

	location_map::const_iterator itLocation = m_locations.find(location.to<lean::utf8_string>());

	if (itLocation != m_locations.end() && !itLocation->second.empty())
		result = itLocation->second.front();

	if (bThrow && result.empty())
		LEAN_THROW_ERROR_CTX("Location unknown", location.c_str());

	return result;
}

// Searches for the given file or directory in the given virtual location.
LEAN_INLINE beCore::Exchange::utf8_string beCore::FileSystem::Impl::Search(const lean::utf8_ntri &location, const lean::utf8_ntri &file, bool bThrow) const
{
	Exchange::utf8_string result;

	location_map::const_iterator itLocation = m_locations.find(location.to<lean::utf8_string>());

	if (itLocation != m_locations.end())
	{
		const path_list &paths = itLocation->second;

		for (path_list::const_iterator itPath = paths.begin(); itPath != paths.end(); ++itPath)
		{
			lean::utf8_string potentialResult = lean::absolute_path<lean::utf8_string>(file, *itPath);

			if (lean::file_exists(potentialResult))
			{
				result.assign(potentialResult.begin(), potentialResult.end());
				break;
			}
		}
	}

	if (result.empty())
	{
		lean::utf8_string potentialResult = lean::absolute_path<lean::utf8_string>(file, location);

		if (lean::file_exists(potentialResult))
			result.assign(potentialResult.begin(), potentialResult.end());
	}

	if (bThrow && result.empty())
		LEAN_THROW_ERROR_XCTX("Fild not found in location", file.c_str(), location.c_str());

	return result;
}

// Shortens the given path, returning a path relative to the given location, if possible.
LEAN_INLINE beCore::Exchange::utf8_string beCore::FileSystem::Impl::Shorten(const lean::utf8_ntri &location, const lean::utf8_ntri &file, bool *pMatch) const
{
	Exchange::utf8_string result;
	
	location_map::const_iterator itLocation = m_locations.find(location.to<lean::utf8_string>());

	if (itLocation != m_locations.end())
	{
		const path_list &paths = itLocation->second;
		size_t maxLength = 0;

		for (path_list::const_iterator it = paths.begin(); it != paths.end(); ++it)
			if (lean::contains_path(file, *it) && it->size() > maxLength)
			{
				bool bMatch = false;

				Exchange::utf8_string relativeFile = lean::relative_path<Exchange::utf8_string>(
						file,
						lean::absolute_path<Exchange::utf8_string>(*it),
						true, &bMatch
					);

				if (bMatch)
				{
					result = relativeFile;
					maxLength = it->size();
				}
			}
	}

	if (pMatch)
		*pMatch = !result.empty();

	if (result.empty())
		result.assign(file.begin(), file.end());

	return result;
}

// Loads a configuration from the given node.
LEAN_INLINE void beCore::FileSystem::Impl::LoadConfiguration(const rapidxml::xml_node<lean::utf8_t> &node)
{
	m_locations.clear();

	for (const rapidxml::xml_node<lean::utf8_t> *locationNode = node.first_node("location");
		locationNode; locationNode = locationNode->next_sibling("location"))
	{
		path_list &paths = m_locations[lean::get_attribute<utf8_t>(*locationNode, "name").to<utf8_string>()];

		for (const rapidxml::xml_node<lean::utf8_t> *pathNode = locationNode->first_node("path");
			pathNode; pathNode = pathNode->next_sibling("path"))
			paths.push_back( lean::canonical_path<path_type>(pathNode->value()) );
	}
}

// Saves the current configuration to the given node.
LEAN_INLINE void beCore::FileSystem::Impl::SaveConfiguration(rapidxml::xml_node<lean::utf8_t> &node) const
{
	rapidxml::xml_document<lean::utf8_t> &document = *node.document();

	for (location_map::const_iterator itLocation = m_locations.begin(); itLocation != m_locations.end(); ++itLocation)
	{
		rapidxml::xml_node<lean::utf8_t> &locationNode = *lean::allocate_node<utf8_t>(document, "location");
		lean::append_attribute<utf8_t>(document, locationNode, "name", itLocation->first);

		const path_list &paths = itLocation->second;

		for (path_list::const_iterator itPath = paths.begin(); itPath != paths.end(); ++itPath)
		{
			rapidxml::xml_node<lean::utf8_t> &pathNode = *lean::allocate_node<utf8_t>(document, "path", *itPath);
			locationNode.append_node(&pathNode);
		}

		node.append_node(&locationNode);
	}
}
