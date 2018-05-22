#pragma once
#include "Mesh/Geometry.h"

namespace Mesh
{
	class Geometry;

	/// A mesh with binding to the pipeline. Includes geometry and the input layout.
	class Bound
	{
		Geometry::P geometry;
		ID3D11InputLayout* inputLayout;

		Bound(Geometry::P geometry, ID3D11InputLayout* inputLayout);
	public:

		/// Shared pointer to type.
		typedef boost::shared_ptr<Bound> P;
		/// Invokes constructor, wraps pointer into shared pointer.
		static Bound::P make(Geometry::P geometry, ID3D11InputLayout* inputLayout) { return Bound::P(new Bound(geometry, inputLayout));}

		~Bound(void);

		/// Set input layout and renders geometry.
		void draw(ID3D11DeviceContext* context);
	};

} // namespace Mesh
