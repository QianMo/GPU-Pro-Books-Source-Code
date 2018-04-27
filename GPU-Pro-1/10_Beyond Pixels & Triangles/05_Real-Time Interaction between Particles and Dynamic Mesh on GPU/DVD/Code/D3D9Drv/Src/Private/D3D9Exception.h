#ifndef D3D9DRV_D3D9EXCEPTION_H_INCLUDED
#define D3D9DRV_D3D9EXCEPTION_H_INCLUDED

#include "ModException.h"

#define D3D9_THROW_IF(expr) { if( (expr) != D3D_OK ) { MD_FERROR( L#expr L" failed!"); } }

#endif