/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_RESOURCE_INDEX
#define BE_CORE_RESOURCE_INDEX

#include "beCore.h"
#include <map>
#include <list>
#include <lean/tags/transitive_ptr.h>
#include <lean/logging/errors.h>

#include <algorithm>
#include <lean/io/numeric.h>

namespace beCore
{

/// Resource index.
template <class Resource, class Info>
class ResourceIndex
{
public:
	/// Value type.
	typedef Resource Resource;
	/// Info type.
	typedef Info Info;

private:
	/// Resource entry.
	struct Entry
	{
		const utf8_string *name;
		const utf8_string *file;

		Info info;

		/// Constructor.
		template <class InfoFW>
		Entry(const utf8_string *name, const utf8_string *file, InfoFW LEAN_FW_REF info)
			: name( name ),
			file( file ),
			info( LEAN_FORWARD(InfoFW, info) ) { }
#ifdef LEAN0X_NEED_EXPLICIT_MOVE
		/// Constructor.
		Entry(Entry &&right)
			: name( right.name ),
			file( right.file ),
			info( std::move(right.info) ) { }
#endif
	};

	typedef std::list<Entry> info_t;
	info_t m_info;

	typedef std::map<Resource*, typename info_t::iterator> resource_map;
	resource_map m_byResource;

	typedef std::map<utf8_string, typename info_t::iterator> string_map;
	string_map m_byName;
	string_map m_byFile;

	template <class Iterator, class Derived>
	class basic_iterator
	{
		friend class ResourceIndex;

	protected:
		Iterator it;

		LEAN_INLINE basic_iterator() { }
		LEAN_INLINE basic_iterator(Iterator it)
			: it(it) { }

	public:
		LEAN_INLINE Derived& operator ++() { ++it; return static_cast<Derived&>(*this); }
		LEAN_INLINE Derived& operator --() { --it; return static_cast<Derived&>(*this); }
		LEAN_INLINE Derived operator ++(int) { Derived prev(it); ++(*this); return prev; }
		LEAN_INLINE Derived operator --(int) { Derived prev(it); --(*this); return prev; }

		template <class It>
		LEAN_INLINE bool operator ==(const It &right) const { return (it == right.it); }
		template <class It>
		LEAN_INLINE bool operator !=(const It &right) const { return (it != right.it); }
	};

	enum iterator_tag { name_tag, file_tag };

	template <class Iterator, class Info>
	class resource_iterator : public basic_iterator< Iterator, resource_iterator<Iterator, Info> >
	{
		friend class ResourceIndex;

	private:
		typedef typename resource_iterator::basic_iterator base_type;
		
		LEAN_INLINE resource_iterator(Iterator it)
			: base_type(it) { }

	public:
		LEAN_INLINE resource_iterator() { }
		template <class OtherIterator, class OtherInfo>
		LEAN_INLINE resource_iterator(const resource_iterator<OtherIterator, OtherInfo> &right)
			: base_type(right.it) { }

		LEAN_INLINE Info& operator *() const { return this->it->info; }
		LEAN_INLINE Info* operator ->() const { return &this->it->info; }

//		LEAN_INLINE Resource* resource() const { return *this->it->resource; }
//		LEAN_INLINE Resource* key() const { return *this->it->resource; }
		LEAN_INLINE Info& value() const { return this->it->info; }
	};

	template <class Iterator, class Info, iterator_tag Tag>
	class string_iterator : public basic_iterator< Iterator, string_iterator<Iterator, Info, Tag> >
	{
		friend class ResourceIndex;

	private:
		typedef typename string_iterator::basic_iterator  base_type;
		
		LEAN_INLINE string_iterator(Iterator it)
			: base_type(it) { }

	public:
		LEAN_INLINE string_iterator() { }
		template <class OtherIterator, class OtherInfo>
		LEAN_INLINE string_iterator(const string_iterator<OtherIterator, OtherInfo, Tag> &right)
			: base_type(right.it) { }

		LEAN_INLINE Info& operator *() const { return this->it->second->info; }
		LEAN_INLINE Info* operator ->() const { return &this->it->second->info; }
		
		template <class OtherIterator>
		LEAN_INLINE operator resource_iterator<OtherIterator, Info>() const { return this->it->second; } 

//		LEAN_INLINE Resource* resource() const { return *this->it->second->resource; }
		LEAN_INLINE utf8_ntr key() const { return utf8_ntr(this->it->first); }
		LEAN_INLINE Info& value() const { return this->it->second->info; }
	};

public:
	/// Unordered resource iterator type.
	typedef resource_iterator<typename info_t::iterator, Info> iterator;
	/// Unordered constant resource iterator type.
	typedef resource_iterator<typename info_t::const_iterator, const Info> const_iterator;
	/// Ordered resource iterator type.
	typedef string_iterator<typename string_map::iterator, Info, name_tag> name_iterator;
	/// Ordered resource iterator type.
	typedef string_iterator<typename string_map::const_iterator, const Info, name_tag> const_name_iterator;
	/// Ordered resource iterator type.
	typedef string_iterator<typename string_map::iterator, Info, file_tag> file_iterator;
	/// Ordered resource iterator type.
	typedef string_iterator<typename string_map::const_iterator, const Info, file_tag> const_file_iterator;

	/// Adds the given resource.
	template <class InfoFW>
	iterator Insert(Resource *resource, const utf8_ntri &name, InfoFW LEAN_FW_REF info)
	{
		typename info_t::iterator it;

		// Name must be valid
		if (name.empty())
			LEAN_THROW_ERROR_MSG("Empty string is not a valid resource name");

		// Try to insert NEW name link
		typename string_map::iterator itByName = m_byName.insert(
				typename string_map::value_type(name.to<utf8_string>(), m_info.end())
			).first;

		// Names must be unique
		if (itByName->second != m_info.end())
			LEAN_THROW_ERROR_CTX("Resource name already taken by another resource", name.c_str());

		try
		{
			/// Try to insert NEW resource link
			typename resource_map::iterator itByResource = m_byResource.insert(
					typename resource_map::value_type(resource, m_info.end())
				).first;

			// Do not re-insert resources
			if (itByResource->second != m_info.end())
				// TODO: Actually a programming error? Assert instead of throwing?
				LEAN_THROW_ERROR_CTX("Resource has been inserted before", name.c_str());

			try
			{
				// Try to insert NEW resource info block
				// ORDER: Map insertions revertible more easily?
				it = m_info.insert( m_info.end(), Entry(&itByName->first, nullptr, LEAN_FORWARD(InfoFW, info)) );

				// ORDER: Establish mapping after successful insertion
				itByName->second = it;
				itByResource->second = it;
			}
			catch (...)
			{
				// NOTE: Never forget to release resource on failure
				if (itByResource->second != m_info.end())
					m_byResource.erase(itByResource);

				throw;
			}
		}
		catch (...)
		{
			// NOTE: Never forget to release name on failure
			if (itByName->second != m_info.end())
				m_byName.erase(itByName);

			throw;
		}

		return it;
	}

	/// Adds the given name to the given resource.
	name_iterator AddName(iterator where, const utf8_ntri &name)
	{
		// Establish one-way mapping
		typename string_map::iterator itByName = m_byName.insert(
				typename string_map::value_type(name.to<utf8_string>(), where.it)
			).first;

		// Names must be unique
		if (itByName->second != where.it)
			LEAN_THROW_ERROR_CTX("Resource name already taken by another resource", name.c_str());

		return itByName;
	}
	
	/// Changes the name of the given resource.
	name_iterator SetName(iterator where, const utf8_ntri &name, bool bKeepOldName = false, bool *pNameChanged = nullptr)
	{
		const utf8_string *oldName = where.it->name;
		typename string_map::iterator itByNewName = AddName(where, name).it;

		bool bNameChange = (&itByNewName->first != oldName);

		// Ignore redundant calls
		if (bNameChange)
		{
			// Establish two-way mapping
			where.it->name = &itByNewName->first;

			// Release old name
			if (!bKeepOldName)
				m_byName.erase(*oldName);
		}

		if (pNameChanged)
			*pNameChanged = bNameChange;

		return itByNewName;
	}

	/// Changes the file of the given resource.
	file_iterator SetFile(iterator where, const utf8_ntri &file, bool *pFileChanged = nullptr, iterator *pUnfiled = nullptr)
	{
		const utf8_string *pOldFile = where.it->file;

		// Try to insert NEW file link
		typename string_map::iterator itByFile = m_byFile.insert(
				typename string_map::value_type(file.to<utf8_string>(), m_info.end())
			).first;

		bool bFileChange = (&itByFile->first != pOldFile);

		if (pUnfiled)
			*pUnfiled = m_info.end();

		// Ignore redundant calls
		if (bFileChange)
		{
			// Unlink previous resource, if necessary
			if (itByFile->second != m_info.end())
			{
				itByFile->second->file = nullptr;

				if (pUnfiled)
					*pUnfiled = itByFile->second;
			}

			// Establish two-way mapping
			itByFile->second = where.it;
			where.it->file = &itByFile->first;

			// Release old file
			if (pOldFile)
				m_byFile.erase(*pOldFile);
		}

		if (pFileChanged)
			*pFileChanged = bFileChange;

		return itByFile;
	}

	/// Unsets the file of the given resource.
	bool Unfile(iterator where)
	{
		const utf8_string *pOldFile = where.it->file;

		if (pOldFile)
		{
			where.it->file = nullptr;
			m_byFile.erase(*pOldFile);
			return true;
		}
		else
			return false;
	}

	/// Links the given new resource to the given iterator.
	iterator Link(iterator where, Resource *resource)
	{
		// Establish one-way mapping
		typename resource_map::iterator itByResource = m_byResource.insert(
				typename resource_map::value_type(resource, where.it)
			).first;

		// Do not re-insert resources
		if (itByResource->second != where.it)
			// TODO: Actually a programming error? Assert instead of throwing?
			LEAN_THROW_ERROR_MSG("Resource has been inserted before");

		return where;
	}
	/// Unlinks the given resource.
	iterator Unlink(iterator where, Resource *resource)
	{
		// Remove one-way mapping
		typename resource_map::iterator itByResource = m_byResource.find(resource);

		if (itByResource != m_byResource.end() && itByResource->second == where.it)
			m_byResource.erase(itByResource);

		return where;
	}

	/// Gets the name of the resource pointed to by the given iterator.
	utf8_ntr GetName(const_iterator where) const
	{
		return utf8_ntr(*where.it->name);
	}
	/// Gets the file of the resource pointed to by the given iterator.
	utf8_ntr GetFile(const_iterator where) const
	{
		return (where.it->file) ? utf8_ntr(*where.it->file) : utf8_ntr("");
	}

	template <class It>
	static std::reverse_iterator<It> reverse_it(It it) { return std::reverse_iterator<It>(it); }

	/// Gets a unique name.
	utf8_string GetUniqueName(const utf8_ntri &name) const
	{
		utf8_string unique;
		uint4 uniqueIdx = 1;

		const utf8_ntri::const_iterator itNumBegin = std::find(reverse_it(name.end()), reverse_it(name.begin()), '.').base();
		const size_t nameLength = (itNumBegin < name.end() - 1) && (lean::char_to_int(itNumBegin + 1, name.end(), uniqueIdx) == name.end())
			? itNumBegin - name.begin()
			: name.size();

		const size_t maxLength = nameLength + lean::max_int_string_length<uint4>::value + 1;

		// Try unaltered name first
		unique = name.to<utf8_string>();

		// Increment index unto name unique
		while (m_byName.find(unique) != m_byName.end())
		{
			unique.resize(maxLength);
			
			utf8_string::iterator idxBegin = unique.begin() + nameLength;
			
			// Append ".#"
			*idxBegin++ = '.';
			unique.erase(
					lean::int_to_char(idxBegin, uniqueIdx++),
					unique.end()
				);
		}

		return unique;
	}

	/// Gets an iterator to the given resource, if existent.
	iterator Find(const Resource *resource) { typename resource_map::iterator it = m_byResource.find(const_cast<Resource*>(resource)); return (it != m_byResource.end()) ? it->second : m_info.end(); }
	/// Gets an iterator to the given resource, if existent.
	const_iterator Find(const Resource *resource) const { typename resource_map::const_iterator it = m_byResource.find(const_cast<Resource*>(resource)); return (it != m_byResource.end()) ? it->second : m_info.end(); }
	/// Gets an iterator to the given resource, if existent.
	name_iterator FindByName(const utf8_string &name) { return m_byName.find(name); }
	/// Gets an iterator to the given resource, if existent.
	const_name_iterator FindByName(const utf8_string &name) const { return m_byName.find(name); }
	/// Gets an iterator to the given resource, if existent.
	file_iterator FindByFile(const utf8_string &file) { return m_byFile.find(file); }
	/// Gets an iterator to the given resource, if existent.
	const_file_iterator FindByFile(const utf8_string &file) const { return m_byFile.find(file); }

	/// Gets an iterator to the lower name bound.
	name_iterator LowerBoundByName(const utf8_string &name) { return m_byName.lower_bound(name); }
	/// Gets an iterator to the lower name bound.
	const_name_iterator LowerBoundByName(const utf8_string &name) const { return m_byName.lower_bound(name); }
	/// Gets an iterator to the upper name bound.
	name_iterator UpperBoundByName(const utf8_string &name) { return m_byName.upper_bound(name); }
	/// Gets an iterator to the upper name bound.
	const_name_iterator UpperBoundByName(const utf8_string &name) const { return m_byName.upper_bound(name); }

	/// Gets an iterator to the lower file bound.
	file_iterator LowerBoundByFile(const utf8_string &file) { return m_byFile.lower_bound(file); }
	/// Gets an iterator to the lower file bound.
	const_file_iterator LowerBoundByFile(const utf8_string &file) const { return m_byFile.lower_bound(file); }
	/// Gets an iterator to the upper file bound.
	file_iterator UpperBoundByFile(const utf8_string &file) { return m_byFile.upper_bound(file); }
	/// Gets an iterator to the upper file bound.
	const_file_iterator UpperBoundByFile(const utf8_string &file) const { return m_byFile.upper_bound(file); }

	/// Gets an iterator to the first resource.
	LEAN_INLINE iterator Begin() { return m_info.begin(); }
	/// Gets an iterator to the first resource.
	LEAN_INLINE const_iterator Begin() const { return m_info.begin(); }
	/// Gets an iterator one past the last resource.
	LEAN_INLINE iterator End() { return m_info.end(); }
	/// Gets an iterator one past the last resource.
	LEAN_INLINE const_iterator End() const { return m_info.end(); }

	/// Gets an iterator to the first resource by name.
	LEAN_INLINE name_iterator BeginByName() { return m_byName.begin(); }
	/// Gets an iterator to the first resource by name.
	LEAN_INLINE const_name_iterator BeginByName() const { return m_byName.begin(); }
	/// Gets an iterator one past the last resource by name.
	LEAN_INLINE name_iterator EndByName() { return m_byName.end(); }
	/// Gets an iterator one past the last resource by name.
	LEAN_INLINE const_name_iterator EndByName() const { return m_byName.end(); }

	/// Gets an iterator to the first resource by file.
	LEAN_INLINE file_iterator BeginByFile() { return m_byFile.begin(); }
	/// Gets an iterator to the first resource by file.
	LEAN_INLINE const_file_iterator BeginByFile() const { return m_byFile.begin(); }
	/// Gets an iterator one past the last resource by file.
	LEAN_INLINE file_iterator EndByFile() { return m_byFile.end(); }
	/// Gets an iterator one past the last resource by file.
	LEAN_INLINE const_file_iterator EndByFile() const { return m_byFile.end(); }

	/// Gets the number of resources.
	LEAN_INLINE uint4 Count() const { return static_cast<uint4>(m_byName.size()); }
};

} // namespace

#endif