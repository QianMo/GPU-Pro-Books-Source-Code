/*******************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*******************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/DX/beError.h"
#include "DXErr.h"

namespace beGraphics
{

namespace DX
{

// Gets an error message describing the given DirectX error.
const utf8_t* GetDXError(HRESULT error)
{
	return ::DXGetErrorStringA(error);
}

} // namespace

} // namespace