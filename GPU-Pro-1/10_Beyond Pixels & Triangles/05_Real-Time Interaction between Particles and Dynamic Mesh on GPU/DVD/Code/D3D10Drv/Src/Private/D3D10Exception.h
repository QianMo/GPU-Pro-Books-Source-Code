#ifndef D3D10DRV_D3D10EXCEPTION_H_INCLUDED
#define D3D10DRV_D3D10EXCEPTION_H_INCLUDED

#include "ModException.h"

#define D3D10_THROW_IF(expr) { if( (expr) != S_OK ) { MD_FERROR( L#expr L" failed!"); } }

#endif