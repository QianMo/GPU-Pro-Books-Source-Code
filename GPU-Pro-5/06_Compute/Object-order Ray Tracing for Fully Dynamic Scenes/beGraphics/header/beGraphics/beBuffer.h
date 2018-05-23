/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_BUFFER
#define BE_GRAPHICS_BUFFER

#include "beGraphics.h"
#include <beCore/beShared.h>

namespace beGraphics
{

/// Buffer wrapper interface.
class Buffer : public beCore::OptionalResource, public Implementation
{
protected:
	LEAN_INLINE Buffer& operator =(const Buffer&) { return *this; }

public:
	virtual ~Buffer() throw() { };
};

} // namespace

#endif