/*******************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*******************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/beAdapters.h"
#include "beGraphics/Any/beAdapters.h"

namespace beGraphics
{

// Creates a graphics object.
lean::resource_ptr<Graphics, true> GetGraphics()
{
	return lean::bind_resource<Graphics>( new Any::Graphics() );
}

} // namespace