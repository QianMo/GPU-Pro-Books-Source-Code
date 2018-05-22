#pragma once

#include "Mesh/Geometry.h"

struct aiMesh;

namespace Mesh
{
	class Geometry;

	class Importer
	{
	public:
		/// Extracts geometry from Open Asset Importer scene.
		static Mesh::Geometry::P fromAiMesh(ID3D11Device* device, aiMesh* assMesh);
	};

} // namespace Mesh