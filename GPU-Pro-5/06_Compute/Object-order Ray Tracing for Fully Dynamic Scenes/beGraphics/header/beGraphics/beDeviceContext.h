/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_DEVICE_CONTEXT
#define BE_GRAPHICS_DEVICE_CONTEXT

#include "beGraphics.h"
#include <beCore/beShared.h>

namespace beGraphics
{

/// Device context interface.
class DeviceContext : public beCore::Shared, public Implementation
{
protected:
	LEAN_INLINE DeviceContext& operator =(const DeviceContext&) { return *this; }

public:
	virtual ~DeviceContext() throw() { }

	/// Clears all state.
	virtual void ClearState() = 0;
};

} // namespace

#endif