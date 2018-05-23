/*****************************************************/
/* lean IO                      (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_IO_WIN_TYPES
#define LEAN_IO_WIN_TYPES

#include "../pimpl/opaque_val.h"

namespace lean
{
namespace io
{

/// @typedef windows_file_handle Opaque windows file handle.
DECLARE_OPAQUE_TYPE(windows_file_handle, void*);
#ifdef _WINDOWS_
DEFINE_OPAQUE_TYPE(windows_file_handle, HANDLE);
#endif

}

using io::windows_file_handle;

}

#endif