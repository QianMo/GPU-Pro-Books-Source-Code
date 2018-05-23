/*****************************************************/
/* lean IO                      (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_IO_IO
#define LEAN_IO_IO

namespace lean
{
	/// Provides I/O utilities such as intrinsic endian conversion, serialization and file classes, etc.
	namespace io { }
}

#include "endianness.h"
#include "numeric.h"
#include "wcharcvt.h"
#include "file.h"
#include "raw_file.h"
#include "raw_file_inserter.h"
#include "mapped_file.h"

#endif