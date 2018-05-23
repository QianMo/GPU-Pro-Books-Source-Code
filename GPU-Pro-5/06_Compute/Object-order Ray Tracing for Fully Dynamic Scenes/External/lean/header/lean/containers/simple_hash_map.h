/*****************************************************/
/* lean Containers              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONTAINERS_SIMPLE_HASH_MAP
#define LEAN_CONTAINERS_SIMPLE_HASH_MAP

#include "../lean.h"
#include "../limits.h"
#include "../smart/terminate_guard.h"
#include "../tags/noncopyable.h"
#include "../functional/hashing.h"
#include "../meta/type_traits.h"
#include <memory>
#include <utility>
#include <cmath>
#include <functional>
#include <string>
#include <iterator>

namespace lean 
{
namespace containers
{

/// Defines construction policies for the class simple_hash_map.
namespace simple_hash_map_policies
{
	/// Simple hash map element construction policy.
	template <bool RawMove = false, bool NoDestruct = false, bool RawKeyMove = RawMove, bool NoKeyDestruct = NoDestruct>
	struct policy
	{
		/// Specifies whether memory containing constructed elements may be moved as a whole, without invoking the contained elements' copy or move constructors.
		static const bool raw_move = RawMove;
		/// Specifies whether memory containing constructed elements may be freed as a whole, without invoking the contained elements' destructors.
		static const bool no_destruct = NoDestruct;
		/// Specifies whether memory containing constructed keys may be moved as a whole, without invoking the contained keys' copy or move constructors.
		static const bool raw_key_move = RawKeyMove;
		/// Specifies whether memory containing constructed keys may be freed as a whole, without invoking the contained keys' destructors.
		static const bool no_key_destruct = NoKeyDestruct;
	};

	/// Default element construction policy.
	typedef policy<> nonpod;
	/// Semi-POD key construction policy (raw move, yet proper destruction).
	typedef policy<false, false, true> semipodkey;
	/// Semi-POD element construction policy (raw move, yet proper destruction).
	typedef policy<true> semipod;
	/// POD key construction policy.
	typedef policy<false, false, true, true> podkey;
	/// POD key / Semi-POD element construction policy.
	typedef policy<true, false, true, true> podkey_semipod;
	/// POD element construction policy.
	typedef policy<true, true> pod;
}

/// Defines default values for invalid & end keys.
template <class Key>
struct default_keys
{
	/// Invalid key value that is guaranteed never to be used in key-value-pairs.
	static const Key invalid_key;
	/// Valid key value used as end marker. May still be used in actual key-value-pairs.
	static const Key end_key;
	/// Predicate used in key validity checks.
	LEAN_INLINE static bool is_valid(const Key &key)
	{
		return (key != invalid_key);
	}
};

// Numeric (generic) defaults
template <class Key>
const Key default_keys<Key>::invalid_key =
	(numeric_limits<Key>::is_unsigned)
		? numeric_limits<Key>::max
		: numeric_limits<Key>::min;
template <class Key>
const Key default_keys<Key>::end_key = Key();

// Pointer defaults
template <class Value>
struct default_keys<Value*>
{
	static Value* const invalid_key;
	static Value* const end_key;

	LEAN_INLINE static bool is_valid(Value *ptr)
	{
		return (ptr != nullptr);
	}
};

template <class Value>
Value* const default_keys<Value*>::invalid_key = nullptr;
template <class Value>
Value* const default_keys<Value*>::end_key = reinterpret_cast<Value*>( static_cast<uintptr_t>(-1) );

// String defaults
template <class Char, class Traits, class Allocator>
struct default_keys< std::basic_string<Char, Traits, Allocator> >
{
	typedef std::basic_string<Char, Traits, Allocator> key_type;

	static const key_type invalid_key;
	static const key_type end_key;

	LEAN_INLINE static bool is_valid(const key_type &key)
	{
		return !key.empty();
	}
};

template <class Char, class Traits, class Allocator>
const std::basic_string<Char, Traits, Allocator>
	default_keys< std::basic_string<Char, Traits, Allocator> >::invalid_key = std::basic_string<Char, Traits, Allocator>();
template <class Char, class Traits, class Allocator>
const std::basic_string<Char, Traits, Allocator>
	default_keys< std::basic_string<Char, Traits, Allocator> >::end_key = std::basic_string<Char, Traits, Allocator>(1, Char());

namespace impl
{

/// Gets the first prime number available that is greater than or equal to the given capacity,
/// may only return a prime number smaller than the given capacity when the actual result would
/// be greater than the given maximum value.
LEAN_MAYBE_EXPORT size_t next_prime_capacity(size_t capacity, size_t max);

/// Simple hash map base.
template < class Key, class Element,
	class Policy,
	class KeyValues,
	class Allocator >
class simple_hash_map_base
{
protected:
	typedef std::pair<const Key, Element> value_type_;

	typedef typename Allocator::template rebind<value_type_>::other allocator_type_;
	allocator_type_ m_allocator;

	value_type_ *m_elements;
	value_type_ *m_elementsEnd;

	typedef typename allocator_type_::size_type size_type_;
	size_type_ m_count;
	size_type_ m_capacity;

	float m_maxLoadFactor;

	static const size_type_ s_maxElementCount = static_cast<size_type_>(-1) / sizeof(value_type_);
	// Make sure size_type is unsigned
	LEAN_STATIC_ASSERT(is_unsigned<size_type_>::value);
	// Reserve end element to allow for proper iteration termination
	static const size_type_ s_maxBucketCount = s_maxElementCount - 1U;
	// Keep one slot open at all times to simplify wrapped find loop termination
	static const size_type_ s_maxSize = s_maxBucketCount - 1U;
	static const size_type_ s_minSize = (32U < s_maxSize) ? 32U : s_maxSize;

	/// Gets the number of buckets required from the given capacity.
	LEAN_INLINE size_type_ buckets_from_capacity(size_type_ capacity)
	{
		LEAN_ASSERT(capacity <= s_maxSize);

		float bucketHint = ceil(capacity / m_maxLoadFactor);
		
		size_type_ bucketCount = (bucketHint >= s_maxBucketCount)
			? s_maxBucketCount
			: static_cast<size_type_>(bucketHint);

		// Keep one slot open at all times to simplify wrapped find loop termination conditions
		return max(bucketCount, capacity + 1U);
	}
	/// Gets the capacity from the given number of buckets.
	LEAN_INLINE size_type_ capacity_from_buckets(size_type_ buckets, size_type_ minCapacity)
	{
		LEAN_ASSERT(buckets <= s_maxBucketCount);
		LEAN_ASSERT(minCapacity <= s_maxSize);

		// Keep one slot open at all times to simplify wrapped find loop termination conditions
		LEAN_ASSERT(minCapacity < buckets);

		return max(
				// Keep one slot open at all times to simplify wrapped find loop termination conditions
				// -> Unsigned overflow handles buckets == 0
				min(static_cast<size_type_>(buckets * m_maxLoadFactor), buckets - 1U),
				// Guarantee minimum capacity
				minCapacity);
	}

	/// Checks whether the given element is physically contained by this hash map.
	LEAN_INLINE bool contains_element(const value_type_ *element) const
	{
		return (m_elements <= element) && (element < m_elementsEnd);
	}
	/// Checks whether the given element is physically contained by this hash map.
	LEAN_INLINE bool contains_element(const value_type_ &element) const
	{
		return contains_element(lean::addressof(element));
	}

	/// Marks the given element as end element.
	static LEAN_INLINE void mark_end(value_type_ *dest)
	{
		new( const_cast<void*>(static_cast<const void*>(lean::addressof(dest->first))) ) Key(KeyValues::end_key);
	}
	/// Invalidates the given element.
	static LEAN_INLINE void invalidate(value_type_ *dest)
	{
		new( const_cast<void*>(static_cast<const void*>(lean::addressof(dest->first))) ) Key(KeyValues::invalid_key);
	}
	/// Invalidates the elements in the given range.
	static void invalidate(value_type_ *dest, value_type_ *destEnd)
	{
		value_type_ *destr = dest;

		try
		{
			for (; dest != destEnd; ++dest)
				invalidate(dest);
		}
		catch (...)
		{
			destruct_keys(destr, dest);
			throw;
		}
	}
	/// Prepares the given element for actual data storage.
	static LEAN_INLINE void revalidate(value_type_ *dest)
	{
		destruct_key(dest);
	}

	/// Destructs only the key of the given element.
	static LEAN_INLINE void destruct_key(value_type_ *destr)
	{
		if (!Policy::no_key_destruct)
			destr->first.~Key();
	}
	/// Destructs the keys of the elements in the given range.
	static void destruct_keys(value_type_ *destr, value_type_ *destrEnd)
	{
		if (!Policy::no_key_destruct)
			for (; destr != destrEnd; ++destr)
				destruct_key(destr);
	}

	/// Helper class that moves element construction exception handling
	/// into a destructor, resulting in less code being generated and
	/// providing automated handling of unexpected invalidation exceptions.
	class invalidate_guard : public noncopyable
	{
	private:
		value_type_ *m_dest;
		bool m_armed;

	public:
		/// Stores an element to be invalidated on destruction, if not disarmed.
		LEAN_INLINE explicit invalidate_guard(value_type_ *dest, bool armed = true)
			: m_dest(dest),
			m_armed(armed) { }
		/// Destructs the stored element, of not disarmed.
		LEAN_INLINE ~invalidate_guard()
		{
			if (m_armed)
				invalidate(m_dest);
		}
		///  Disarms this guard.
		LEAN_INLINE void disarm() { m_armed = false; }
	};

	/// Helper class that moves element destruction exception handling
	/// into a destructor, resulting in less code being generated and
	/// providing automated handling of unexpected invalidation exceptions.
	class invalidate_n_guard : public noncopyable
	{
	private:
		value_type_ *m_dest;
		value_type_ *m_destEnd;
		bool m_armed;

	public:
		/// Stores elements to be invalidated on destruction, if not disarmed.
		LEAN_INLINE explicit invalidate_n_guard(value_type_ *dest, value_type_ *destEnd, bool armed = true)
			: m_dest(dest),
			m_destEnd(destEnd),
			m_armed(armed) { }
		/// Destructs the stored elements, of not disarmed.
		LEAN_INLINE ~invalidate_n_guard()
		{
			if (m_armed)
				invalidate(m_dest, m_destEnd);
		}
		///  Disarms this guard.
		LEAN_INLINE void disarm() { m_armed = false; }
	};

	/// Default constructs an element at the given location.
	LEAN_INLINE void default_construct(value_type_ *dest, const Key &key)
	{
		invalidate_guard guard(dest);

		revalidate(dest);
		m_allocator.construct(dest, value_type_(key, Element()));

		guard.disarm();
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Default constructs an element at the given location.
	LEAN_INLINE void default_construct(value_type_ *dest, Key &&key)
	{
		invalidate_guard guard(dest);

		revalidate(dest);
		m_allocator.construct(dest, value_type_(std::move(key), Element()));

		guard.disarm();
	}
#endif
	/// Copies the given source element to the given destination.
	LEAN_INLINE void copy_construct(value_type_ *dest, const value_type_ &source)
	{
		invalidate_guard guard(dest);

		revalidate(dest);
		m_allocator.construct(dest, source);

		guard.disarm();
	}
	/// Moves the given source element to the given destination.
	LEAN_INLINE void move_construct(value_type_ *dest, value_type_ &source)
	{
#ifndef LEAN0X_NO_RVALUE_REFERENCES
		invalidate_guard guard(dest);
		
		revalidate(dest);
		m_allocator.construct(dest, std::move(source));
		
		guard.disarm();
#else
		copy_construct(dest, source);
#endif
	}
	/// Moves the given source element to the given destination.
	LEAN_INLINE void move(value_type_ *dest, value_type_ &source)
	{
#ifndef LEAN0X_NO_RVALUE_REFERENCES
		const_cast<Key&>(dest->first) = std::move(const_cast<Key&>(source.first));
		dest->second = std::move(source.second);
#else
		const_cast<Key&>(dest->first) = const_cast<Key&>(source.first);
		dest->second = source.second;
#endif
	}
	
	/// Destructs the given VALID element.
	LEAN_INLINE void destruct_element(value_type_ *destr)
	{
		if (!Policy::no_destruct)
			m_allocator.destroy(destr);
	}

	/// Destructs both valid and invalid elements in the given range.
	void destruct(value_type_ *destr, value_type_ *destrEnd)
	{
		if (!Policy::no_destruct || !Policy::no_key_destruct)
			for (; destr != destrEnd; ++destr)
				if (key_valid(destr->first))
					destruct_element(destr);
				else
					destruct_key(destr);
	}

	/// Destructs and invalidates valid elements in the given range.
	void destruct_and_invalidate(value_type_ *destr, value_type_ *destrEnd)
	{
		// Elements invalidated (overwritten) by guard on exception
		invalidate_n_guard invalidateGuard(destr, destrEnd);

		for (; destr != destrEnd; ++destr)
			if (key_valid(destr->first))
			{
				// Don't handle exceptions explicitly, resources leaking anyways
				destruct_element(destr);
				invalidate(destr);
			}

		// Everything has been invalidated correctly
		invalidateGuard.disarm();
	}

	/// Triggers a length error.
	LEAN_NOINLINE static void length_exceeded()
	{
		throw std::length_error("simple_hash_map<K, E> too long");
	}
	/// Checks the given length.
	LEAN_INLINE static void check_length(size_type_ count)
	{
		if (count > s_maxSize)
			length_exceeded();
	}

	/// Initializes the this hash map base.
	LEAN_INLINE explicit simple_hash_map_base(float maxLoadFactor)
		: m_elements(nullptr),
		m_elementsEnd(nullptr),
		m_count(0),
		m_capacity(0),
		m_maxLoadFactor(maxLoadFactor) { }
	/// Initializes the this hash map base.
	LEAN_INLINE simple_hash_map_base(float maxLoadFactor, const allocator_type_ &allocator)
		: m_allocator(allocator),
		m_elements(nullptr),
		m_elementsEnd(nullptr),
		m_count(0),
		m_capacity(0),
		m_maxLoadFactor(maxLoadFactor) { }
	/// Initializes the this hash map base.
	LEAN_INLINE simple_hash_map_base(const simple_hash_map_base &right)
		: m_allocator(right.m_allocator),
		m_elements(nullptr),
		m_elementsEnd(nullptr),
		m_count(0),
		m_capacity(0),
		m_maxLoadFactor(right.m_maxLoadFactor) { }
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Initializes the this hash map base.
	LEAN_INLINE simple_hash_map_base(simple_hash_map_base &&right) noexcept
		: m_allocator(std::move(right.m_allocator)),
		m_elements(std::move(right.m_elements)),
		m_elementsEnd(std::move(right.m_elementsEnd)),
		m_count(std::move(right.m_count)),
		m_capacity(std::move(right.m_capacity)),
		m_maxLoadFactor(std::move(right.m_maxLoadFactor)) { }
#endif

	/// Does nothing.
	LEAN_INLINE simple_hash_map_base& operator =(const simple_hash_map_base&) { return *this; }
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Does nothing.
	LEAN_INLINE simple_hash_map_base& operator =(simple_hash_map_base&&) noexcept { return *this; }
#endif

	/// Returns true if the given key is valid.
	LEAN_INLINE static bool key_valid(const Key &key) { return KeyValues::is_valid(key); }

	/// Swaps the contents of this hash map base and the given hash map base.
	LEAN_INLINE void swap(simple_hash_map_base &right) noexcept
	{
		using std::swap;

		swap(m_allocator, right.m_allocator);
		swap(m_elements, right.m_elements);
		swap(m_elementsEnd, right.m_elementsEnd);
		swap(m_count, right.m_count);
		swap(m_capacity, right.m_capacity);
		swap(m_maxLoadFactor, right.m_maxLoadFactor);
	}
};

}

/// Simple and fast hash map class, partially implementing the STL hash map interface.
template < class Key, class Element,
	class Policy = simple_hash_map_policies::nonpod,
	class Hash = hash<Key>,
	class KeyValues = default_keys<Key>,
    class Pred = std::equal_to<Key>,
	class Allocator = std::allocator<Element> >
class simple_hash_map : private impl::simple_hash_map_base<Key, Element, Policy, KeyValues, Allocator>
{
private:
	typedef impl::simple_hash_map_base<Key, Element, Policy, KeyValues, Allocator> base_type;

	typedef Hash hasher_;
	hasher_ m_hasher;
	typedef Pred key_equal_;
	key_equal_ m_keyEqual;

	/// Allocates space for the given number of elements.
	void reallocate(size_type_ newBucketCount, size_type_ minCapacity = 0U)
	{
		// Make prime (required for universal modulo hashing)
		newBucketCount = impl::next_prime_capacity(newBucketCount, s_maxBucketCount);
		
		// Guarantee minimum capacity
		// ASSERT: One slot always remains open, automatically terminating find loops
		if (newBucketCount <= minCapacity)
			length_exceeded();
		
		// Use end element to allow for proper iteration termination
		const size_type_ newElementCount = newBucketCount + 1U;
		
		value_type_ *newElements = m_allocator.allocate(newElementCount);
		value_type_ *newElementsEnd = newElements + newBucketCount;
		
		try
		{
			// ASSERT: End element key always valid to allow for proper iteration termination
			mark_end(newElementsEnd);

			try
			{
				invalidate(newElements, newElementsEnd);
			}
			catch(...)
			{
				destruct_key(newElementsEnd);
				throw;
			}
			
			if (!empty())
				try
				{
					// ASSERT: One slot always remains open, automatically terminating find loops
					LEAN_ASSERT(size() < newBucketCount);

					for (value_type_ *element = m_elements; element != m_elementsEnd; ++element)
						if (base_type::key_valid(element->first))
							move_construct(
								locate_element(element->first, newElements, newElementsEnd, newBucketCount).second,
								*element );
				}
				catch(...)
				{
					destruct(newElements, newElementsEnd);
					destruct_key(newElementsEnd);
					throw;
				}
		}
		catch(...)
		{
			m_allocator.deallocate(newElements, newElementCount);
			throw;
		}
		
		value_type_ *oldElements = m_elements;
		value_type_ *oldElementsEnd = m_elementsEnd;
		const size_type_ oldBucketCount = bucket_count();
		
		m_elements = newElements;
		m_elementsEnd = newElementsEnd;
		m_capacity = capacity_from_buckets(newBucketCount, minCapacity);
		
		if (oldElements)
			free(oldElements, oldElementsEnd, oldBucketCount + 1U);
	}
	
	/// Frees all elements.
	LEAN_INLINE void free()
	{
		if (m_elements)
			free(m_elements, m_elementsEnd, bucket_count() + 1U);
	}
	/// Frees the given elements.
	LEAN_INLINE void free(value_type_ *elements, value_type_ *elementsEnd, size_type_ elementCount)
	{
		// ASSERT: End element key always valid to allow for proper iteration termination

		// Do nothing on exception, resources leaking anyways!
		destruct(elements, elementsEnd);
		destruct_key(elementsEnd);
		m_allocator.deallocate(elements, elementCount);
	}

	/// Gets the first element that might contain the given key.
	LEAN_INLINE value_type_* first_element(const Key &key) const
	{
		return first_element(key, m_elements, bucket_count());
	}
	/// Gets the first element that might contain the given key.
	LEAN_INLINE value_type_* first_element(const Key &key, value_type_ *elements, size_type_ bucketCount) const
	{
		return elements + m_hasher(key) % bucketCount;
	}
	/// Gets the element stored under the given key and returns false if existent, otherwise returns true and gets a fitting open element slot.
	LEAN_INLINE std::pair<bool, value_type_*> locate_element(const Key &key) const
	{
		return locate_element(key, m_elements, m_elementsEnd, bucket_count());
	}
	/// Gets the element stored under the given key and returns false if existent, otherwise returns true and gets a fitting open element slot.
	LEAN_INLINE std::pair<bool, value_type_*> locate_element(const Key &key, value_type_ *elements, value_type_ *elementsEnd, size_type_ bucketCount) const
	{
		LEAN_ASSERT(base_type::key_valid(key));

		value_type_ *element = first_element(key, elements, bucketCount);

		while (base_type::key_valid(element->first))
		{
			if (m_keyEqual(element->first, key))
				return std::make_pair(false, element);
			
			// Wrap around
			if (++element == elementsEnd)
				element = elements;

			// ASSERT: One slot always remains open, automatically terminating this loop
		}

		return std::make_pair(true, element);
	}
	/// Gets the element stored under the given key, if existent, returns end otherwise.
	LEAN_INLINE value_type_* find_element(const Key &key) const
	{
		value_type_ *element = first_element(key);
		
		while (base_type::key_valid(element->first))
		{
			if (m_keyEqual(element->first, key))
				return element;
			
			// Wrap around
			if (++element == m_elementsEnd)
				element = m_elements;

			// ASSERT: One slot always remains open, automatically terminating this loop
		}

		return m_elementsEnd;
	}
	/// Removes the element stored at the given location.
	LEAN_INLINE void remove_element(value_type_ *element)
	{
		// If anything goes wrong, we won't be able to fix it
		terminate_guard terminateGuard;

		value_type_ *hole = element;
		
		// Wrap around
		if (++element == m_elementsEnd)
			element = m_elements;

		// Find next empty position
		while (base_type::key_valid(element->first))
		{
			value_type_ *auxElement = first_element(element->first);
			
			bool tooLate = (auxElement <= hole);
			bool tooEarly = (element < auxElement);
			bool wrong = (hole <= element) ? (tooLate || tooEarly) : (tooLate && tooEarly);
			
			// Move wrongly positioned elements into hole
			if (wrong)
			{
				move(hole, *element);
				hole = element;
			}
			
			// Wrap around
			if (++element == m_elementsEnd)
				element = m_elements;

			// ASSERT: One slot always remains open, automatically terminating this loop
		}

		destruct_element(hole);
		invalidate(hole);
		--m_count;

		terminateGuard.disarm();
	}
	/// Copies all elements from the given hash map into this _empty_ hash map of sufficient capacity.
	LEAN_INLINE void copy_elements_to_empty(const value_type_ *elements, const value_type_ *elementsEnd)
	{
		LEAN_ASSERT(empty());

		for (const value_type *element = elements; element != elementsEnd; ++element)
			if (base_type::key_valid(element->first))
			{
				copy_construct(
					locate_element(element->first).second,
					*element );
				++m_count;
			}
	}

	/// Grows hash map storage to fit the given new count.
	LEAN_INLINE void growTo(size_type_ newCount, bool checkLength = true)
	{
		// Mind overflow
		if (checkLength)
			check_length(newCount);

		reallocate(buckets_from_capacity(next_capacity_hint(newCount)), newCount);
	}
	/// Grows hash map storage to fit the given additional number of elements.
	LEAN_INLINE void grow(size_type_ count)
	{
		size_type_ oldSize = size();

		// Mind overflow
		if (count > s_maxSize || s_maxSize - count < oldSize)
			length_exceeded();

		growTo(oldSize + count, false);
	}
	/// Grows hash map storage to fit the given new count, not inlined.
	LEAN_NOINLINE void growToHL(size_type_ newCount)
	{
		growTo(newCount);
	}
	/// Grows hash map storage to fit the given additional number of elements, not inlined.
	LEAN_NOINLINE void growHL(size_type_ count)
	{
		grow(count);
	}

public:
	/// Construction policy used.
	typedef Policy construction_policy;

	/// Type of the allocator used by this hash map.
	typedef allocator_type_ allocator_type;
	/// Type of the size returned by this hash map.
	typedef size_type_ size_type;
	/// Type of the difference between the addresses of two elements in this hash map.
	typedef typename allocator_type::difference_type difference_type;

	/// Type of pointers to the elements contained by this hash map.
	typedef typename allocator_type::pointer pointer;
	/// Type of constant pointers to the elements contained by this hash map.
	typedef typename allocator_type::const_pointer const_pointer;
	/// Type of references to the elements contained by this hash map.
	typedef typename allocator_type::reference reference;
	/// Type of constant references to the elements contained by this hash map.
	typedef typename allocator_type::const_reference const_reference;
	/// Type of the elements contained by this hash map.
	typedef typename allocator_type::value_type value_type;
	/// Type of the keys stored by this hash map.
	typedef Key key_type;
	/// Type of the elements contained by this hash map.
	typedef Element mapped_type;

	/// Simple hash map iterator class.
	template <class Element>
	class basic_iterator
	{
	friend class simple_hash_map;

	private:
		Element *m_element;

		/// Allows for the automated validation of iterators on construction.
		enum search_first_valid_t
		{
			/// Allows for the automated validation of iterators on construction.
			search_first_valid
		};

		/// Constructs an iterator from the given element.
		LEAN_INLINE basic_iterator(Element *element)
			: m_element(element) { }
		/// Constructs an iterator from the given element or the next valid element, should the current element prove invalid.
		LEAN_INLINE basic_iterator(Element *element, search_first_valid_t)
			: m_element(
				(element && !base_type::key_valid(element->first))
					? (++basic_iterator(element)).m_element
					: element
				) { }

	public:
		/// Iterator category.
		typedef std::forward_iterator_tag iterator_category;
		/// Type of the difference between the addresses of two elements in this hash map.
		typedef typename simple_hash_map::difference_type difference_type;
		/// Type of the values iterated.
		typedef Element value_type;
		/// Type of references to the values iterated.
		typedef value_type& reference;
		/// Type of pointers to the values iterated.
		typedef value_type* pointer;

		/// Gets the current element.
		LEAN_INLINE reference operator *() const
		{
			return *m_element;
		}
		/// Gets the current element.
		LEAN_INLINE pointer operator ->() const
		{
			return m_element;
		}

		/// Continues iteration.
		LEAN_INLINE basic_iterator& operator ++()
		{
			do
			{
				++m_element;
			}
			// ASSERT: End element key is always valid
			while (!base_type::key_valid(m_element->first));

			return *this;
		}
		/// Continues iteration.
		LEAN_INLINE basic_iterator operator ++(int)
		{
			basic_iterator prev(*this);
			++(*this);
			return prev;
		}

		/// Comparison operator.
		LEAN_INLINE bool operator ==(const basic_iterator &right) const
		{
			return (m_element == right.m_element);
		}
		/// Comparison operator.
		LEAN_INLINE bool operator !=(const basic_iterator &right) const
		{
			return (m_element != right.m_element);
		}
	};

	/// Type of iterators to the elements contained by this hash map.
	typedef basic_iterator<value_type> iterator;
	/// Type of constant iterators to the elements contained by this hash map.
	typedef basic_iterator<const value_type> const_iterator;
	/// Type of iterators to the elements contained by this hash map.
	typedef iterator local_iterator;
	/// Type of constant iterators to the elements contained by this hash map.
	typedef const_iterator const_local_iterator;

	/// Type of the hash function.
	typedef hasher_ hasher;
	/// Type of the key comparison function.
	typedef key_equal_ key_equal;

	/// Constructs an empty hash map.
	simple_hash_map()
		: base_type(0.75f)
	{
		LEAN_ASSERT(key_valid(KeyValues::end_key));
	}
	/// Constructs an empty hash map.
	explicit simple_hash_map(size_type capacity, float maxLoadFactor = 0.75f)
		: base_type(maxLoadFactor)
	{
		LEAN_ASSERT(key_valid(KeyValues::end_key));
		growTo(capacity);
	}
	/// Constructs an empty hash map.
	simple_hash_map(size_type capacity, float maxLoadFactor, const hasher& hash)
		: base_type(maxLoadFactor),
		m_hasher(hash)
	{
		LEAN_ASSERT(key_valid(KeyValues::end_key));
		growTo(capacity);
	}
	/// Constructs an empty hash map.
	simple_hash_map(size_type capacity, float maxLoadFactor, const hasher& hash, const key_equal& keyComp)
		: base_type(maxLoadFactor),
		m_hasher(hash),
		m_keyEqual(keyComp)
	{
		LEAN_ASSERT(key_valid(KeyValues::end_key));
		growTo(capacity);
	}
	/// Constructs an empty hash map.
	simple_hash_map(size_type capacity, float maxLoadFactor, const hasher& hash, const key_equal& keyComp, const allocator_type &allocator)
		: base_type(maxLoadFactor, allocator),
		m_hasher(hash),
		m_keyEqual(keyComp)
	{
		LEAN_ASSERT(key_valid(KeyValues::end_key));
		growTo(capacity);
	}
	/// Copies all elements from the given hash map to this hash map.
	simple_hash_map(const simple_hash_map &right)
		: base_type(right),
		m_hasher(right.m_hasher),
		m_keyEqual(right.m_keyEqual)
	{
		if (!right.empty())
		{
			growTo(right.size());

			try
			{
				copy_elements_to_empty(right.m_elements, right.m_elementsEnd);
			}
			catch (...)
			{
				free();
				throw;
			}
		}
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Moves all elements from the given hash map to this hash map.
	simple_hash_map(simple_hash_map &&right) noexcept
		: base_type(std::move(right)),
		m_hasher(std::move(right.m_hasher)),
		m_keyEqual(std::move(right.m_keyEqual))
	{
		right.m_elements = nullptr;
		right.m_elementsEnd = nullptr;
		right.m_count = 0;
		right.m_capacity = 0;
	}
#endif
	/// Destroys all elements in this hash map.
	~simple_hash_map()
	{
		free();
	}

	/// Copies all elements of the given hash map to this hash map.
	simple_hash_map& operator =(const simple_hash_map &right)
	{
		if (&right != this)
		{
			// Clear before reallocation to prevent full-range moves
			clear();

			if (!right.empty())
			{
				if (right.size() > capacity())
					growToHL(right.size());
				
				copy_elements_to_empty(right.m_elements, right.m_elementsEnd);
			}
		}
		return *this;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Moves all elements from the given hash map to this hash map.
	simple_hash_map& operator =(simple_hash_map &&right) noexcept
	{
		if (&right != this)
		{
			free();

			m_elements = std::move(right.m_elements);
			m_elementsEnd = std::move(right.m_elementsEnd);

			right.m_elements = nullptr;
			right.m_elementsEnd = nullptr;

			m_allocator = std::move(right.m_allocator);
		}
		return *this;
	}
#endif

	/// Inserts a default-constructed value into the hash map using the given key, if none
	/// stored under the given key yet, otherwise returns the one currently stored.
	LEAN_INLINE reference insert(const key_type &key)
	{
		LEAN_ASSERT(base_type::key_valid(key));

		if (m_count == capacity())
			growHL(1);

		std::pair<bool, value_type*> element = locate_element(key);
		
		if (element.first)
		{
			default_construct(element.second, key);
			++m_count;
		}
		return *element.second;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Inserts a default-constructed value into the hash map using the given key, if none
	/// stored under the given key yet, otherwise returns the one currently stored.
	LEAN_INLINE reference insert(key_type &&key)
	{
		LEAN_ASSERT(base_type::key_valid(key));

		if (m_count == capacity())
			growHL(1);

		std::pair<bool, value_type*> element = locate_element(key);
		
		if (element.first)
		{
			default_construct(element.second, std::move(key));
			++m_count;
		}
		return *element.second;
	}
#endif
	/// Inserts the given key-value-pair into this hash map.
	LEAN_INLINE std::pair<iterator, bool> insert(const value_type &value)
	{
		LEAN_ASSERT(base_type::key_valid(value.first));

		if (m_count == capacity())
		{
			if (contains_element(value))
				return std::make_pair(iterator(const_cast<value_type*>(lean::addressof(value))), false);

			growHL(1);
		}

		std::pair<bool, value_type*> element = locate_element(value.first);

		if (element.first)
		{
			copy_construct(element.second, value);
			++m_count;
		}
		return std::make_pair(iterator(element.second), element.first);
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Inserts the given key-value-pair into this hash map.
	LEAN_INLINE std::pair<iterator, bool> insert(value_type &&value) // wrong way round
	{
		LEAN_ASSERT(base_type::key_valid(value.first));

		if (m_count == capacity())
		{
			if (contains_element(value))
				return std::make_pair(iterator(lean::addressof(value)), false);

			growHL(1);
		}

		std::pair<bool, value_type*> element = locate_element(value.first);

		if (element.first)
		{
			move_construct(element.second, value);
			++m_count;
		}
		return std::make_pair(iterator(element.second), element.first);
	}
#endif
	/// Removes the element stored under the given key, if any.
	LEAN_INLINE size_type erase(const key_type &key)
	{
		// Explicitly handle unallocated state
		value_type* element = (!empty())
			? find_element(key)
			: m_elementsEnd;

		if (element != m_elementsEnd)
		{
			remove_element(element);
			return 1;
		}
		else
			return 0;
	}
	/// Removes the element that the given iterator is pointing to.
	LEAN_INLINE iterator erase(iterator where)
	{
		LEAN_ASSERT(contains_element(where.m_element));
		
		remove_element(where.m_element);
		return iterator(where.m_element, iterator::search_first_valid);
	}

	/// Clears all elements from this hash map.
	LEAN_INLINE void clear()
	{
		m_count = 0;
		destruct_and_invalidate(m_elements, m_elementsEnd);
	}

	/// Reserves space for the predicted number of elements given.
	LEAN_INLINE void reserve(size_type newCapacity)
	{
		// Mind overflow
		check_length(newCapacity);

		if (newCapacity > capacity())
			reallocate(buckets_from_capacity(newCapacity), newCapacity);
	}
	/// Tries to grow or shrink the hash map to fit the given number of elements given.
	/// The hash map will never shrink below the number of elements currently stored.
	LEAN_INLINE void rehash(size_type newCapacity)
	{
		newCapacity = max(size(), newCapacity);

		// Mind overflow
		check_length(newCapacity);

		if (newCapacity != capacity())
			reallocate(buckets_from_capacity(newCapacity), newCapacity);
	}

	/// Gets an element by key, returning end() on failure.
	LEAN_INLINE iterator find(const key_type &key) { return (!empty()) ? iterator(find_element(key)) : end(); }
	/// Gets an element by key, returning end() on failure.
	LEAN_INLINE const_iterator find(const key_type &key) const { return (!empty()) ? const_iterator(find_element(key)) : end(); }

	/// Gets an element by key, inserts a new default-constructed one if none existent yet.
	LEAN_INLINE mapped_type& operator [](const key_type &key) { return insert(key).second; }
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Gets an element by key, inserts a new default-constructed one if none existent yet.
	LEAN_INLINE mapped_type& operator [](key_type &&key) { return insert(std::move(key)).second; }
#endif

	/// Returns an iterator to the first element contained by this hash map.
	LEAN_INLINE iterator begin(void) { return iterator(m_elements, iterator::search_first_valid); }
	/// Returns a constant iterator to the first element contained by this hash map.
	LEAN_INLINE const_iterator begin(void) const { return const_iterator(m_elements, const_iterator::search_first_valid); }
	/// Returns an iterator beyond the last element contained by this hash map.
	LEAN_INLINE iterator end(void) { return iterator(m_elementsEnd); }
	/// Returns a constant iterator beyond the last element contained by this hash map.
	LEAN_INLINE const_iterator end(void) const { return const_iterator(m_elementsEnd); }

	/// Gets a copy of the allocator used by this hash map.
	LEAN_INLINE allocator_type get_allocator() const { return m_allocator; };

	/// Returns true if the given key is valid.
	LEAN_INLINE bool key_valid(const key_type &key) const { return base_type::key_valid(key); }

	/// Returns true if the hash map is empty.
	LEAN_INLINE bool empty(void) const { return (m_count == 0); };
	/// Returns the number of elements contained by this hash map.
	LEAN_INLINE size_type size(void) const { return m_count; };
	/// Returns the number of elements this hash map could contain without reallocation.
	LEAN_INLINE size_type capacity(void) const { return m_capacity; };
	/// Gets the current number of buckets.
	LEAN_INLINE size_type bucket_count() const { return m_elementsEnd - m_elements; }

	/// Gets the maximum load factor.
	LEAN_INLINE float max_load_factor() const { return m_maxLoadFactor; }
	/// Sets the maximum load factor.
	void max_load_factor(float factor)
	{
		m_maxLoadFactor = factor;

		// Make sure capacity never goes below the number of elements currently stored
		// -> Capacity equal count will result in reallocation on next element insertion
		m_capacity = capacity_from_buckets(bucket_count(), size());
	}

	/// Gets the current load factor.
	LEAN_INLINE float load_factor() const { return static_cast<float>(m_count) / static_cast<float>(m_capacity); };

	/// Computes a new capacity based on the given number of elements to be stored.
	size_type next_capacity_hint(size_type count) const
	{
		size_type oldCapacity = capacity();
		LEAN_ASSERT(oldCapacity <= s_maxSize);

		// Try to double capacity (mind overflow)
		size_type newCapacity = (s_maxSize - oldCapacity < oldCapacity)
			? 0
			: oldCapacity + oldCapacity;
		
		if (newCapacity < count)
			newCapacity = count;

		if (newCapacity < s_minSize)
			newCapacity = s_minSize;
		
		return newCapacity;
	}

	/// Swaps the contents of this hash map and the given hash map.
	LEAN_INLINE void swap(simple_hash_map &right) noexcept
	{
		using std::swap;

		swap(m_keyEqual, right.m_keyEqual);
		swap(m_hasher, right.m_hasher);
		base_type::swap(right);
	}
	/// Estimates the maximum number of elements that may be constructed.
	LEAN_INLINE size_type max_size() const
	{
		return s_maxSize;
	}
};

/// Swaps the contents of the given hash maps.
template <class Element, class Policy, class Allocator>
LEAN_INLINE void swap(simple_hash_map<Element, Policy, Allocator> &left, simple_hash_map<Element, Policy, Allocator> &right) noexcept
{
	left.swap(right);
}

} // namespace

namespace simple_hash_map_policies = containers::simple_hash_map_policies;
using containers::simple_hash_map;

} // namespace

#ifdef LEAN_INCLUDE_LINKED
#include "source/simple_hash_map.cpp"
#endif

#endif