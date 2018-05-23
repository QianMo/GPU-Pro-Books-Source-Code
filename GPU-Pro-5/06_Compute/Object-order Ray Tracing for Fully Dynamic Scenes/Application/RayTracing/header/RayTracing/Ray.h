#pragma once

#include "Tracing.h"

#include <beMath/beVectorDef.h>

namespace app
{

namespace tracing
{

struct RayDesc
{
	bem::fvec3 Orig;
	bem::fvec3 Dir;
};

struct VoxelTriangle
{
	uint4 CellID;
	uint4 TriangleID;
};

struct RayVoxelTriangle
{
	uint4 RayID;
	uint4 CellID;
	uint4 TriangleID;
};

} // namespace

} // namespace