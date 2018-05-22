#pragma once
#include "Mesh/Geometry.h"

namespace Mesh
{

	struct VertexStreamDesc
	{
		static const D3D11_INPUT_ELEMENT_DESC defaultElements[2];
		static const float defaultData[20];

		VertexStreamDesc():elements(defaultElements),nElements(2),vertexStride(5*sizeof(float)), nVertices(4), vertexData((void*)defaultData) {}

		const D3D11_INPUT_ELEMENT_DESC* elements;
		unsigned int nElements;
		unsigned int vertexStride;
		unsigned int nVertices;
		void* vertexData;
	};

	/// Geometry description with a single vertex buffer.
	class VertexStream :
		public Mesh::Geometry
	{
		ID3D11Buffer* vertexBuffer;
		D3D11_INPUT_ELEMENT_DESC* elements;
		unsigned int nElements;
		unsigned int vertexStride;
		unsigned int nVertices;
		VertexStream(ID3D11Device* device, VertexStreamDesc& desc);
	public:
		/// Shared pointer type.
		typedef boost::shared_ptr<VertexStream> P;
		/// Shared pointer array type.
		typedef std::vector< VertexStream::P > A;
		/// Invokes contructor, returns shared pointer.
		static VertexStream::P make(ID3D11Device* device, VertexStreamDesc& desc) { return VertexStream::P(new VertexStream(device, desc));}
		~VertexStream(void);

		void getElements(const D3D11_INPUT_ELEMENT_DESC*& elements, unsigned int& nElements);
		unsigned int getElementCount() {return nElements;}

		/// Renders vertex buffer.
		void draw(ID3D11DeviceContext* context);

		unsigned int getStride() {return vertexStride;}
		ID3D11Buffer* getBuffer() {return vertexBuffer;}

	};


} // namespace Mesh