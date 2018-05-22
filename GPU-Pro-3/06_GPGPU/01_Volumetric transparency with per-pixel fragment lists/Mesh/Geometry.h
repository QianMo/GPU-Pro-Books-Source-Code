#pragma once

namespace Mesh
{

	/// Base class for mesh geoemtries.
	class Geometry
	{
	public:
		/// Smart pointer type.
		typedef boost::shared_ptr<Geometry> P;

		virtual ~Geometry(void)
		{
		}

		/// Returns vertex element description.
		virtual void getElements(const D3D11_INPUT_ELEMENT_DESC*& elements, unsigned int& nElements)=0;

		/// Renders geometry.
		virtual void draw(ID3D11DeviceContext* context)=0;

	};

} // namespace Mesh