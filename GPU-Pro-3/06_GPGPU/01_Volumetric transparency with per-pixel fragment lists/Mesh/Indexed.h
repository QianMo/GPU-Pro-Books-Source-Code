#pragma once
#include "Mesh/Geometry.h"
#include "Mesh/VertexStream.h"

namespace Mesh
{
	struct IndexBufferDesc
	{
		static const unsigned short defaultData[6];

		DXGI_FORMAT indexFormat;
		unsigned int nPrimitives;
		D3D11_PRIMITIVE_TOPOLOGY topology;
		unsigned int nIndices;
		void* indexData;

		unsigned int getIndexStride()
		{
			return (indexFormat==DXGI_FORMAT_R16_UINT)?sizeof(unsigned short):sizeof(unsigned int);
		}

		IndexBufferDesc():indexFormat(DXGI_FORMAT_R16_UINT), nPrimitives(2), topology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST), nIndices(0), indexData((void*)defaultData) {}
	};

	class VertexStream;

	/// Geometry with index buffer.
	class Indexed :
		public Mesh::Geometry
	{
		ID3D11Buffer** vertexBuffers;
		unsigned int* vertexStrides;
		unsigned int nVertexBuffers;

		const D3D11_INPUT_ELEMENT_DESC* elements;
		unsigned int nElements;

		ID3D11Buffer* indexBuffer;
	
		DXGI_FORMAT indexFormat;
		unsigned int nPrimitives;
		unsigned int nIndices;

		D3D11_PRIMITIVE_TOPOLOGY topology;

		Indexed(ID3D11Device* device, IndexBufferDesc& desc, VertexStream::A& vertexStreams);
	public:

		/// Shared pointer type.
		typedef boost::shared_ptr<Indexed> P;
		/// Invokes contructor, returns shared pointer. Single vertex buffer version.
		static Indexed::P make(ID3D11Device* device, IndexBufferDesc& desc, VertexStream::P vertexStream) { VertexStream::A arr(1); arr.at(0)=vertexStream; return Indexed::P(new Indexed(device, desc, arr));}
		/// Invokes contructor, returns shared pointer. Multiple vertex buffers version.
		static Indexed::P make(ID3D11Device* device, IndexBufferDesc& desc, VertexStream::A& vertexStreams) { return Indexed::P(new Indexed(device, desc, vertexStreams));}

		~Indexed(void);

		/// Gets vertex element description.
		void getElements(const D3D11_INPUT_ELEMENT_DESC*& elements, unsigned int& nElements);

		/// Renders the geometry.
		void draw(ID3D11DeviceContext* context);

		unsigned int getVertexBufferCount() {return nVertexBuffers;}
		ID3D11Buffer** getVertexBuffers() {return vertexBuffers;}
		unsigned int* getVertexStrides() {return vertexStrides;}

		unsigned int getPrimitiveCount() {return nPrimitives;}
		DXGI_FORMAT getIndexFormat() {return indexFormat;}
		D3D11_PRIMITIVE_TOPOLOGY getTopology() {return topology;}
		ID3D11Buffer* getIndexBuffer() {return indexBuffer;}

		unsigned int getIndexCount() { return nIndices; }

	};

} // namespace Mesh