/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_PERSISTENT_IDS
#define BE_CORE_PERSISTENT_IDS

#include "beCore.h"
#include <lean/tags/noncopyable.h>
#include <typeinfo>
#include <vector>

namespace beCore
{

/// Persistent ID manager.
class PersistentIDs : public lean::noncopyable
{
public:
	/// Reference.
	struct Reference
	{
		uint8 id;
		void *pointer;
		const std::type_info *type;

		/// Constructor.
		Reference(uint8 id, void *pointer, const std::type_info &type)
			: id(id),
			pointer(pointer),
			type(&type) { }
	};

private:
	typedef std::vector<Reference> ref_vector;
	ref_vector m_references;

	uint8 m_nextID;

public:
	/// Invalid ID.
	static const uint8 InvalidID = static_cast<uint8>(-1);

	/// Constructor.
	BE_CORE_API PersistentIDs(uint8 startID = 0);
	/// Destructor.
	BE_CORE_API ~PersistentIDs();

	/// Reserves an ID.
	BE_CORE_API uint8 ReserveID();
	/// Gets the next ID.
	BE_CORE_API uint8 GetNextID() const;
	/// Skips all IDs up to the given the next ID.
	BE_CORE_API void SkipIDs(uint8 nextID);

	/// Adds a new reference.
	BE_CORE_API uint8 AddReference(void *ptr, const std::type_info &type);
	/// Updates a reference.
	BE_CORE_API bool SetReference(uint8 id, void *ptr, const std::type_info &type, bool bNoOverwrite = false);
	/// Gets a reference.
	BE_CORE_API void* GetReference(uint8 id, const std::type_info &type) const;
	/// Unsets a reference.
	BE_CORE_API void UnsetReference(uint8 id, const void *compare = nullptr, bool bErase = true);

	/// Adds a new reference.
	template <class Type>
	LEAN_INLINE uint8 AddReference(Type *ptr)
	{
		return AddReference(ptr, typeid(Type));
	}
	/// Updates a reference.
	template <class Type>
	LEAN_INLINE bool SetReference(uint8 id, Type *ptr, bool bNoOverwrite = false)
	{
		return SetReference(id, ptr, typeid(Type), bNoOverwrite);
	}
	/// Gets a reference.
	template <class Type>
	LEAN_INLINE Type* GetReference(uint8 id) const
	{
		return static_cast<Type*>( GetReference(id, typeid(Type)) );
	}
};

} // namespace

#endif