/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_ANY_API
#define BE_GRAPHICS_ANY_API

#include "beGraphics.h"

#ifdef BE_GRAPHICS_DIRECTX_11
	#include "../DX11/beD3D11.h"
#endif

namespace beGraphics
{
	namespace API = Any::API;
	namespace api = Any::api;
}

#endif