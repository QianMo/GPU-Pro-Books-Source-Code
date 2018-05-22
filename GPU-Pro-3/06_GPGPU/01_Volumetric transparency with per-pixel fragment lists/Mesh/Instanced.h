#pragma once
#include "geometry.h"
#include "Mesh/Indexed.h"


namespace Mesh
{
	struct InstanceBufferDesc
	{
		static const D3D11_INPUT_ELEMENT_DESC defaultElements[3];
		static const D3DXVECTOR4 defaultData[3];
	
		InstanceBufferDesc():elements(defaultElements),nElements(3),instanceStride(12*sizeof(float)), instanceData((void*)defaultData) {}

		const D3D11_INPUT_ELEMENT_DESC* elements;
		unsigned int nElements;
		unsigned int instanceStride;
		void* instanceData;
	};

	class Indexed;

	/// Geometry with instancing information.
	class Instanced :
		public Mesh::Geometry
	{
		ID3D11Buffer** vertexBuffers;
		unsigned int* vertexStrides;
		unsigned int nVertexBuffers;
		unsigned int nIndexedVertexBuffers;

		const D3D11_INPUT_ELEMENT_DESC* elements;
		unsigned int nElements;

		ID3D11Buffer* indexBuffer;
	
		DXGI_FORMAT indexFormat;
		unsigned int nIndices;
		unsigned int nPrimitives;

		D3D11_PRIMITIVE_TOPOLOGY topology;

		unsigned int nInstances;
		Instanced(ID3D11Device* device, unsigned int nInstances, InstanceBufferDesc* desc, unsigned int nInstanceBuffers, Mesh::Indexed::P indexed);
	public:
		/// Shared pointer type.
		typedef boost::shared_ptr<Instanced> P;
		/// Invokes contructor, returns shared pointer.
		static Instanced::P make(ID3D11Device* device, unsigned int nInstances, InstanceBufferDesc* desc, unsigned int nInstanceBuffers, Mesh::Indexed::P indexed) { return Instanced::P(new Instanced(device, nInstances, desc, nInstanceBuffers, indexed));}
	
		~Instanced(void);

		/// Gets vertex element description.
		void getElements(const D3D11_INPUT_ELEMENT_DESC*& elements, unsigned int& nElements);
		/// Renders geometry with instancing.
		void draw(ID3D11DeviceContext* context);

		ID3D11Buffer* getVertexBuffer(unsigned int iVertexBuffer) { return vertexBuffers[iVertexBuffer];}
		ID3D11Buffer* getInstanceBuffer(unsigned int iInstanceBuffer) { return vertexBuffers[iInstanceBuffer + nIndexedVertexBuffers];}
	};

} // namespace Mesh