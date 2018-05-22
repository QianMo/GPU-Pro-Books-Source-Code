#pragma once
#include "d3dx11effect.h"
#include "Mesh/Geometry.h"
#include "Mesh/Bound.h"


namespace Mesh
{

	class Geometry;
	class Bound;

	/// Manages input layouts.
	class Binder
	{
		class InputConfiguration
		{
			friend class Mesh::Binder;

			const D3D11_INPUT_ELEMENT_DESC* elements;
			unsigned int nElements;
			const D3DX11_PASS_DESC& passDesc;

			ID3D11InputLayout* inputLayout;

			InputConfiguration(const D3DX11_PASS_DESC& passDesc, Mesh::Geometry::P geometry);

			/// Returns true if input signatures are identical and elements with shared semantics are also identical.
			bool isCompatible(const InputConfiguration& other) const;

			HRESULT createInputLayout(ID3D11Device* device);
		public:
			~InputConfiguration();
		};

		typedef std::vector<InputConfiguration*> InputConfigurationList;
		InputConfigurationList inputConfigurationList;

		ID3D11InputLayout* getCompatibleInputLayout(ID3D11Device* device, const D3DX11_PASS_DESC& passDesc, Mesh::Geometry::P geometry);

	public:
		Binder();
		~Binder();

		/// Creates a mesh with binding to the pipeline, caching the input layout.
		Bound::P bind(ID3D11Device* device, const D3DX11_PASS_DESC& passDesc, Mesh::Geometry::P geometry);
	};

} // namespace Mesh