#pragma once

#include "Mesh/Geometry.h"

namespace Mesh
{
	class Geometry;

	class GeometryLoader
	{
		static Mesh::Geometry::P createGeometryFromMemory(ID3D11Device* device, BYTE* data, unsigned int nBytes);

	public:
		/// Loads geometry from simple custom binary format (.dgb or .gbs)
		static Mesh::Geometry::P createGeometryFromFile(ID3D11Device* device, const char* filename);
	};

} // namespace Mesh