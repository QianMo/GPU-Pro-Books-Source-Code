#pragma once

// Master header
#include <beCore/beCore.h>
#include <string>

namespace app
{

using namespace breeze;
using namespace lean::types;
LEAN_REIMPORT_NUMERIC_TYPES;
using namespace lean::strings::types;

std::string identityString(void const* id);

} // namespace

// Right now, we depend on CUDA for sorting
#define CUDA_ENABLED
#define BENCHMARK_TIMING