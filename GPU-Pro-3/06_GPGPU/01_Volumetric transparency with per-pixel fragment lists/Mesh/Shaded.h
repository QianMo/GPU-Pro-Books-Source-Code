#pragma once
#include "Mesh/Bound.h"
#include "Mesh/Material.h"

namespace Mesh
{
	/// Mesh with material. Handles every setting necessary for drawing the mesh.
	class Shaded
	{
		Bound::P bound;
		Material::P material;

		Shaded(Bound::P bound, Material::P material):bound(bound), material(material){}
	public:
		/// Shared pointer type.
		typedef boost::shared_ptr<Shaded> P;
		/// Invokes contructor, returns shared pointer.
		static Shaded::P make(Bound::P bound, Material::P material) { return Shaded::P(new Shaded(bound, material));}

		/// Renders mesh.
		void draw(ID3D11DeviceContext* context);
	};

} // namespace Mesh
