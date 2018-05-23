/*****************************************************/
/* lean Smart                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_SMART_CLONEABLE
#define LEAN_SMART_CLONEABLE

#include "../lean.h"

namespace lean
{
namespace smart
{

/// Base class that may be used to tag a specific class cloneable.
class LEAN_INTERFACE cloneable
{
	LEAN_INTERFACE_BEHAVIOR(cloneable)

public:
	/// Constructs and returns a clone of this cloneable.
	virtual cloneable* clone() const = 0;
	/// Moves the contents of this cloneable to a clone.
	virtual cloneable* clone_move() { return clone(); }
	/// Destroys a clone.
	virtual void destroy() const = 0;
};

/// Clones the given cloneable.
LEAN_INLINE cloneable* clone(const cloneable &cln) { return cln.clone(); }
#ifndef LEAN0X_NO_RVALUE_REFERENCES
/// Clones the given cloneable by moving.
LEAN_INLINE cloneable* clone(cloneable &&cln) { return cln.clone_move(); }
#endif
/// Destroys the given cloneable.
LEAN_INLINE void destroy(const cloneable *cln) { if (cln) cln->destroy(); }

} // namespace

using smart::cloneable;
using smart::clone;

} // namespace

#endif