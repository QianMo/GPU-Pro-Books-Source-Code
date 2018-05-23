/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_POOL
#define BE_CORE_POOL

#include "beCore.h"
#include <lean/smart/scoped_ptr.h>
#include <lean/containers/simple_vector.h>

namespace beCore
{

/// Simple object pool template.
template <class Type>
struct Pool
{
public:
	/// Element type.
	typedef Type Element;
	/// Element vector type.
	typedef lean::simple_vector< lean::scoped_ptr<Element>, lean::containers::vector_policies::semipod > ElementVector;
	/// Managed elements.
	ElementVector Elements;

	/// Gets a free element, nullptr if none available.
	Element* FreeElement()
	{
		for (typename ElementVector::iterator it = Elements.begin(); it != Elements.end(); ++it)
		{
			Element *element = it->get();

			if (!element->IsUsed())
				return element;
		}

		return nullptr;
	}
	/// Adds the given element.
	Element* AddElement(Element *element) noexcept
	{
		LEAN_ASSERT_NOT_NULL(element);
		try
		{
			new_emplace(Elements) lean::scoped_ptr<Element>(element);
		}
		LEAN_ASSERT_NOEXCEPT
		return element;
	}
};

} // namespace

#endif