/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PASS_SEQUENCE
#define BE_SCENE_PASS_SEQUENCE

#include "beScene.h"
#include <beCore/beMany.h>

namespace beScene
{

/// Multi-pass effect binder interface.
template <class Pass, class PassIterator = const Pass*>
class LEAN_INTERFACE PassSequence
{
	LEAN_INTERFACE_BEHAVIOR(PassSequence)

public:
	/// Pass type.
	typedef Pass PassType;
	typedef beCore::Range<PassIterator> PassRange;

	/// Gets the pass identified by the given ID.
	virtual PassRange GetPasses() const = 0;
};

} // namespace

#endif