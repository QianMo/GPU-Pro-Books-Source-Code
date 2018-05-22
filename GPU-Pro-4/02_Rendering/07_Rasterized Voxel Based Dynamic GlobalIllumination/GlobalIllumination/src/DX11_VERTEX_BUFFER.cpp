#include <stdafx.h>
#include <DEMO.h>
#include <DX11_VERTEX_BUFFER.h>

const char *formatNames[] = { "float","float2","float3","float4" };
DXGI_FORMAT formats[] = { DXGI_FORMAT_R32_FLOAT,DXGI_FORMAT_R32G32_FLOAT,
                          DXGI_FORMAT_R32G32B32_FLOAT,DXGI_FORMAT_R32G32B32A32_FLOAT }; 
int formatSizes[] = { 1,2,3,4 };
const char *elementNames[] = { "position","texCoords","normal","tangent","color" };
const char *semanticNames[] = { "POSITION","TEXCOORD","NORMAL","TANGENT","COLOR" };

void DX11_VERTEX_BUFFER::Release()
{
  SAFE_RELEASE(vertexBuffer);
	SAFE_RELEASE(inputLayout);
	vertices.Erase();
}

bool DX11_VERTEX_BUFFER::Create(const VERTEX_ELEMENT_DESC *vertexElementDescs,int numVertexElementDescs,bool dynamic,int maxVertexCount)
{	
	this->dynamic = dynamic;
	this->maxVertexCount = maxVertexCount;
	for(int i=0;i<numVertexElementDescs;i++)
		vertexSize += formatSizes[vertexElementDescs[i].format];
	vertices.Init(vertexSize);
	vertices.Resize(maxVertexCount);

	// create tmp shader for input layout
	std::string str;
	str = "struct VS_INPUT\n{\n  ";
	for(int i=0;i<numVertexElementDescs;i++)
	{
		VERTEX_ELEMENT_DESC desc = vertexElementDescs[i];
		str += formatNames[desc.format];
		str += " ";
		str += elementNames[desc.vertexElement];
		str += ": ";
		str += semanticNames[desc.vertexElement];
		str += ";\n  ";
	}
	str +="};\n";
	str += "struct VS_OUTPUT\n{\n  float4 Pos: SV_POSITION;\n};\n";
	str += "VS_OUTPUT main(VS_INPUT input)\n{\n  VS_OUTPUT output = (VS_OUTPUT)0;\n  return output;\n};";

	// compile tmp shader
	ID3DBlob* vsBlob = NULL;
	ID3DBlob* errorBlob = NULL;
	if(D3DX11CompileFromMemory(str.c_str(),str.length(),"tmp",NULL,NULL,"main","vs_5_0",
     D3DCOMPILE_ENABLE_STRICTNESS,0,NULL,&vsBlob,&errorBlob,NULL)!=S_OK)	
	{
		if(errorBlob)
		{
			MessageBox(NULL,(char*)errorBlob->GetBufferPointer(),"Vertex Shader Error",MB_OK|MB_ICONEXCLAMATION);
			SAFE_RELEASE(errorBlob);
		}
		return false;
	}
	SAFE_RELEASE(errorBlob);

	// create input layout
	D3D11_INPUT_ELEMENT_DESC *layout = new D3D11_INPUT_ELEMENT_DESC[numVertexElementDescs];
	for(int i=0;i<numVertexElementDescs;i++)
	{
		VERTEX_ELEMENT_DESC desc = vertexElementDescs[i];
		layout[i].SemanticName = semanticNames[desc.vertexElement];
		layout[i].SemanticIndex = 0;
		layout[i].Format = formats[desc.format];
		layout[i].InputSlot = 0;
		layout[i].AlignedByteOffset = desc.offset*sizeof(float);
		layout[i].InputSlotClass =  D3D11_INPUT_PER_VERTEX_DATA;
		layout[i].InstanceDataStepRate = 0;
	}
	if(DEMO::renderer->GetDevice()->CreateInputLayout(layout,numVertexElementDescs,vsBlob->GetBufferPointer(),
		 vsBlob->GetBufferSize(),&inputLayout)!=S_OK)
	{
		SAFE_RELEASE(vsBlob);
		SAFE_DELETE_ARRAY(layout);
		return false;
	}
	SAFE_RELEASE(vsBlob);
	SAFE_DELETE_ARRAY(layout);

	// Create vertex buffer
	D3D11_BUFFER_DESC bd;
	ZeroMemory(&bd,sizeof(bd));
	bd.ByteWidth = sizeof(float)*vertexSize*maxVertexCount;
	if(dynamic)
	{
		bd.Usage = D3D11_USAGE_DYNAMIC;
		bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	}
	else
	{
		bd.Usage = D3D11_USAGE_DEFAULT;
		bd.CPUAccessFlags = 0;
	}
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	if(DEMO::renderer->GetDevice()->CreateBuffer(&bd,NULL,&vertexBuffer)!=S_OK)
	{
		SAFE_RELEASE(inputLayout);
		return false;
	} 
	return true;
}

int DX11_VERTEX_BUFFER::AddVertices(int numVertices,const float *newVertices)
{
	int firstListIndex = vertices.AddElements(numVertices,newVertices);
  assert(vertices.GetSize()<=maxVertexCount);
	return firstListIndex;
}

bool DX11_VERTEX_BUFFER::Update()
{
	if(vertices.GetSize()>0)
	{
		if(dynamic)
		{
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			if(DEMO::renderer->GetDeviceContext()->Map(vertexBuffer,0,D3D11_MAP_WRITE_DISCARD,0,&mappedResource)!=S_OK)
			  return false;
			memcpy(mappedResource.pData,vertices,vertexSize*sizeof(float)*vertices.GetSize());
			DEMO::renderer->GetDeviceContext()->Unmap(vertexBuffer,0);
		}
		else
		{
			DEMO::renderer->GetDeviceContext()->UpdateSubresource(vertexBuffer,0,NULL,vertices,0,0);
		}
	}
	return true;
}

void DX11_VERTEX_BUFFER::Bind() const
{
	DEMO::renderer->GetDeviceContext()->IASetInputLayout(inputLayout);
	UINT stride = sizeof(float)*vertexSize;
	UINT offset = 0;
	DEMO::renderer->GetDeviceContext()->IASetVertexBuffers(0,1,&vertexBuffer,&stride,&offset);
}
