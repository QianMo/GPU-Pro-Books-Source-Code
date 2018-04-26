/**
 *	Pedro Hermosilla
 *	
 *	Moving Group - UPC
 *	Mesh.cpp
 */

#include <iostream>
#include <fstream>
#include <map>

#include "TriMesh.h"
#include "TriMesh_algo.h"

#include "Mesh.h"

#define FLOAT_COMP 0.00001

#define LENGTH(x,y,z)sqrt((x*x)+(y*y)+(z*z))
#define CROSSPRODUCT(vect1,vect2,res) { \
	res[0] = (vect1[1]*vect2[2])-(vect1[2]*vect2[1]); \
	res[1] = (vect1[2]*vect2[0])-(vect1[0]*vect2[2]); \
	res[2] = (vect1[0]*vect2[1])-(vect1[1]*vect2[0]); }

Mesh::Mesh(ID3D10Device* d3dDevice)
{
	_d3dDevice = d3dDevice;
	_vertexBuffer = NULL;
	_indexBuffer = NULL;
	_adyIndexBuffer = NULL;
	_vertexSize = 0;
	_xMin = 0.0f;
	_yMin = 0.0f;
	_zMin = 0.0f;
	_xMax = 0.0f;
	_yMax = 0.0f;
	_zMax = 0.0f;;
}

Mesh::~Mesh(void)
{
	if(_vertexBuffer)
		_vertexBuffer->Release();
	if(_indexBuffer)
		_indexBuffer->Release();
	if(_adyIndexBuffer)
		_adyIndexBuffer->Release();
}

void Mesh::load(const char* fileName)
{
	std::ifstream iFile;
	iFile.open(fileName,std::ios::in);
	if(iFile.good()){

		char* buffer = new char[256];
		std::vector<float> vertexs;
		std::vector<float> normals;
		std::vector<float> curv;
		std::vector<unsigned int> faces;

		char auxBuff[32];
		ZeroMemory(&auxBuff[0],sizeof(char)*32);
		float auxFloat;
		unsigned int auxUInt;
		unsigned int i = 1;
		unsigned int auxIndex = 0;

		bool initSearch = false;
		
		while(iFile.good())
		{
			iFile.getline(buffer,256);
			switch(buffer[0])
			{
				case 'v':
					switch(buffer[1])
					{
						case ' ':
							ZeroMemory(&auxBuff[0],sizeof(char)*32);
							i = 1;
							auxIndex = 0;
							initSearch = false;

							do
							{
								i++;
								if((buffer[i] == ' ' || buffer[i] == 0) && initSearch)
								{
									auxFloat = atof(&auxBuff[0]);
									vertexs.push_back(auxFloat);
									ZeroMemory(&auxBuff[0],sizeof(char)*32);
									auxIndex = 0;
								}else{
									if(auxIndex < 32)
									{
										auxBuff[auxIndex] = buffer[i];
										auxIndex++;
										initSearch = true;
									}
								}
							}while(buffer[i] != 0);
							break;
						default:
							break;
					}
					break;
				case 'f':
					ZeroMemory(&auxBuff[0],sizeof(char)*32);
					i = 1;
					auxIndex = 0;
					initSearch = false;

					do
					{
						i++;
						if((buffer[i] == ' ' || buffer[i] == 0) && initSearch)
						{
							auxUInt = atoi(&auxBuff[0]) -1;
							faces.push_back(auxUInt);
							ZeroMemory(&auxBuff[0],sizeof(char)*32);
							auxIndex = 0;
						}else{
							if(auxIndex < 32)
							{
								auxBuff[auxIndex] = buffer[i];
								auxIndex++;
								initSearch = true;
							}
						}
					}while(buffer[i] != 0);
					break;
				default:
					break;
			}
		}
		delete[] buffer;
		iFile.close();

		//unifyVertexs(vertexs,faces);

		_numVertexs = vertexs.size()/3;
		_numFaces = faces.size()/3;

		_xMin = vertexs[0];
		_yMin = vertexs[1];
		_zMin = vertexs[2];
		_xMax = vertexs[0];
		_yMax = vertexs[1];
		_zMax = vertexs[2];

		for(unsigned int i = 3; i < vertexs.size(); i+=3)
		{
			if(vertexs[i] < _xMin)
				_xMin = vertexs[i];
			else if(vertexs[i] > _xMax)
				_xMax = vertexs[i];

			if(vertexs[i+1] < _yMin)
				_yMin = vertexs[i+1];
			else if(vertexs[i+1] > _yMin)
				_yMin = vertexs[i+1];

			if(vertexs[i+2] < _zMin)
				_zMin = vertexs[i+2];
			else if(vertexs[i+2] > _zMin)
				_zMin = vertexs[i+2];
		}

		computeNormals(vertexs,normals,faces);

		std::vector<unsigned int> adyFaces;

		computeAdy(vertexs,normals,faces,adyFaces);

		computeCurv(vertexs,normals,curv,faces);

		createBuffers(vertexs,normals,curv,faces,adyFaces);

	}
}

void Mesh::unifyVertexs(std::vector<float>& vertexs,std::vector<unsigned int>& faces)
{
	std::vector<float> auxVertexs;
	std::vector<unsigned int> auxIndexs;
	bool repeated;
	unsigned int j = 0;
	double aux1,aux2,aux3;

	for(unsigned int i = 0; i < vertexs.size(); i+=3)
	{
		repeated = false;
		j = 0;
		while(!repeated && j < auxVertexs.size())
		{
			aux1 = vertexs[i] - auxVertexs[j];
			aux2 = vertexs[i+1] - auxVertexs[j+1];
			aux3 = vertexs[i+2] - auxVertexs[j+2];
			if(aux1 > -FLOAT_COMP && aux1 < FLOAT_COMP &&
			   aux2 > -FLOAT_COMP && aux2 < FLOAT_COMP &&
			   aux3 > -FLOAT_COMP && aux3 < FLOAT_COMP){
				repeated = true;
				auxIndexs.push_back(j/3);
			}

			j+=3;
		}
		if(!repeated){
			auxIndexs.push_back(auxVertexs.size()/3);
			auxVertexs.push_back(vertexs[i]);
			auxVertexs.push_back(vertexs[i+1]);
			auxVertexs.push_back(vertexs[i+2]);
		}
	}

	vertexs.clear();

	for(unsigned int i = 0; i < auxVertexs.size(); i++)
		vertexs.push_back(auxVertexs[i]);

	std::vector<unsigned int> auxIndexs2;
	for(unsigned int i = 0; i < faces.size(); i++)
		auxIndexs2.push_back(auxIndexs[faces[i]]);

	faces.clear();

	for(unsigned int i = 0; i < auxIndexs2.size(); i++)
		faces.push_back(auxIndexs2[i]);
}

void Mesh::computeNormals(std::vector<float>& vertexs, std::vector<float>& normals, 
			std::vector<unsigned int>& faces)
{
	float vert1[3];
	float vert2[3];
	float aux;
	float normal[3];

	normals.reserve(vertexs.size());
	for(unsigned int i = 0; i < vertexs.size(); i++)
		normals.push_back(0.0f);

	for(unsigned int i = 0; i < faces.size(); i+=3)
	{
		vert1[0] = vertexs[faces[i+1]*3] - vertexs[faces[i]*3];
		vert1[1] = vertexs[(faces[i+1]*3)+1] - vertexs[(faces[i]*3)+1];
		vert1[2] = vertexs[(faces[i+1]*3)+2] - vertexs[(faces[i]*3)+2];
		aux = 1.0f/LENGTH(vert1[0],vert1[1],vert1[2]);
		vert1[0] *= aux;
		vert1[1] *= aux;
		vert1[2] *= aux;

		vert2[0] = vertexs[faces[i+2]*3] - vertexs[faces[i+1]*3];
		vert2[1] = vertexs[(faces[i+2]*3)+1] - vertexs[(faces[i+1]*3)+1];
		vert2[2] = vertexs[(faces[i+2]*3)+2] - vertexs[(faces[i+1]*3)+2];
		aux = 1.0f/LENGTH(vert2[0],vert2[1],vert2[2]);
		vert2[0] *= aux;
		vert2[1] *= aux;
		vert2[2] *= aux;

		CROSSPRODUCT(vert1,vert2,normal);
		aux = 1.0f/LENGTH(normal[0],normal[1],normal[2]);
		normal[0] *= aux;
		normal[1] *= aux;
		normal[2] *= aux;

		normals[faces[i]*3] += normal[0];
		normals[(faces[i]*3)+1] += normal[1];
		normals[(faces[i]*3)+2] += normal[2];

		normals[faces[i+1]*3] += normal[0];
		normals[(faces[i+1]*3)+1] += normal[1];
		normals[(faces[i+1]*3)+2] += normal[2];

		normals[faces[i+2]*3] += normal[0];
		normals[(faces[i+2]*3)+1] += normal[1];
		normals[(faces[i+2]*3)+2] += normal[2];

	}

	for(unsigned int i = 0; i < normals.size(); i+=3)
	{
		aux = 1.0f/LENGTH(normals[i],normals[i+1],normals[i+2]);
		normals[i] *= aux;
		normals[i+1] *= aux;
		normals[i+2] *= aux;
	}
}

struct compFunct
{
	bool operator()(std::pair<unsigned int,unsigned int> fP, std::pair<unsigned int,unsigned int> sP) const
	{
		if(fP.first < sP.first){
			return true;
		}else if(fP.first > sP.first){
			return false;
		}else{
			if(fP.second < sP.second)
				return true;
			else
				return false;
		}
	}
};

void Mesh::computeAdy(std::vector<float>& vertexs, std::vector<float>& normals, 
			std::vector<unsigned int>& faces,std::vector<unsigned int>& adyFaces)
{

	std::map<std::pair<unsigned int, unsigned int>,std::pair<int,int>,compFunct> edgeList;
	std::map<std::pair<unsigned int, unsigned int>,std::pair<int,int>,compFunct>::iterator edgeIter;

	std::pair<int,int> auxPair;
	unsigned int index1;
	unsigned int index2;
	unsigned int index3;
	std::pair<unsigned int,unsigned int> key;
	for(unsigned int i = 0; i < faces.size(); i+=3)
	{
		index1 = faces[i];
		index2 = faces[i+1];
		index3 = faces[i+2];

		key.first = (index1 < index2)?index1:index2;
		key.second = (index1 < index2)?index2:index1;
		edgeIter = edgeList.find(key);
		if(edgeIter == edgeList.end()){
			auxPair.first = index3;
			auxPair.second = -1;
			edgeList.insert(std::pair<std::pair<unsigned int,unsigned int>,std::pair<int,int>>(key,auxPair));
		}else{
			(*edgeIter).second.second = index3;
		}

		key.first = (index2 < index3)?index2:index3;
		key.second = (index2 < index3)?index3:index2;
		edgeIter = edgeList.find(key);
		if(edgeIter == edgeList.end()){
			auxPair.first = index1;
			auxPair.second = -1;
			edgeList.insert(std::pair<std::pair<unsigned int,unsigned int>,std::pair<int,int>>(key,auxPair));
		}else{
			(*edgeIter).second.second = index1;
		}

		key.first = (index3 < index1)?index3:index1;
		key.second = (index3 < index1)?index1:index3;
		edgeIter = edgeList.find(key);
		if(edgeIter == edgeList.end()){
			auxPair.first = index2;
			auxPair.second = -1;
			edgeList.insert(std::pair<std::pair<unsigned int,unsigned int>,std::pair<int,int>>(key,auxPair));
		}else{
			(*edgeIter).second.second = index2;
		}
	}

	int auxIndex;

	for(unsigned int i = 0; i < faces.size(); i+=3)
	{
		index1 = faces[i];
		index2 = faces[i+1];
		index3 = faces[i+2];

		adyFaces.push_back(index1);

		key.first = (index1 < index2)?index1:index2;
		key.second = (index1 < index2)?index2:index1;
		edgeIter = edgeList.find(key);
		auxIndex = ((*edgeIter).second.first == index3)?(*edgeIter).second.second:(*edgeIter).second.first;
		adyFaces.push_back((auxIndex > 0)?(unsigned int)auxIndex:index3);
		
		adyFaces.push_back(index2);

		key.first = (index2 < index3)?index2:index3;
		key.second = (index2 < index3)?index3:index2;
		edgeIter = edgeList.find(key);
		auxIndex = ((*edgeIter).second.first == index1)?(*edgeIter).second.second:(*edgeIter).second.first;
		adyFaces.push_back((auxIndex > 0)?(unsigned int)auxIndex:index1);
		
		adyFaces.push_back(index3);

		key.first = (index3 < index1)?index3:index1;
		key.second = (index3 < index1)?index1:index3;
		edgeIter = edgeList.find(key);
		auxIndex = ((*edgeIter).second.first == index2)?(*edgeIter).second.second:(*edgeIter).second.first;
		adyFaces.push_back((auxIndex > 0)?(unsigned int)auxIndex:index2);
		
	}

	edgeList.clear();
}

void Mesh::computeCurv(std::vector<float>& vertexs, std::vector<float>& normals,
			std::vector<float>& curv, std::vector<unsigned int>& faces)
{
	TriMesh* mesh = new TriMesh();
	point auxPoint;
	for(unsigned int i = 0; i < vertexs.size(); i+=3)
	{
		auxPoint[0] = vertexs[i];
		auxPoint[1] = vertexs[i+1];
		auxPoint[2] = vertexs[i+2];
		mesh->vertices.push_back(auxPoint);
	}

	vec auxNormal;
	for(unsigned int i = 0; i < normals.size(); i+=3)
	{
		auxNormal[0] = normals[i];
		auxNormal[1] = normals[i+1];
		auxNormal[2] = normals[i+2];
		mesh->normals.push_back(auxNormal);
	}

	TriMesh::Face auxFace;
	for(unsigned int i = 0; i < (faces.size()/3); i+=3)
	{
		auxFace.v[0] = faces[i];
		auxFace.v[1] = faces[i+1];
		auxFace.v[2] = faces[i+2];
		mesh->faces.push_back(auxFace);
	}
	
	mesh->need_curvatures();

	curv.clear();
	float aux;
	for(unsigned int i = 0; i < mesh->pdir2.size(); i++)
	{
		aux = 1.0f/LENGTH(mesh->pdir2[i][0],mesh->pdir2[i][1],mesh->pdir2[i][2]);
		curv.push_back(mesh->pdir2[i][0]*aux);
		curv.push_back(mesh->pdir2[i][1]*aux);
		curv.push_back(mesh->pdir2[i][2]*aux);
	}

	delete mesh;
}

void Mesh::createBuffers(std::vector<float>& vertexs, std::vector<float>& normals,
	std::vector<float>& curv, std::vector<unsigned int>& faces, std::vector<unsigned int>& adyFaces)
{

	D3D10_INPUT_ELEMENT_DESC descEntry;
	descEntry.SemanticName = "POSITION";
	descEntry.SemanticIndex = 0;
	descEntry.Format = DXGI_FORMAT_R32G32B32_FLOAT;
	descEntry.InputSlot = 0;
	descEntry.AlignedByteOffset = 0;
	descEntry.InputSlotClass = D3D10_INPUT_PER_VERTEX_DATA;
	descEntry.InstanceDataStepRate = 0;
	_vertexDescr.push_back(descEntry);

	_vertexSize = 3;
	int offset = 12;

	descEntry.SemanticName = "NORMAL";
	descEntry.SemanticIndex = 0;
	descEntry.Format = DXGI_FORMAT_R32G32B32_FLOAT;
	descEntry.InputSlot = 0;
	descEntry.AlignedByteOffset = offset;
	descEntry.InputSlotClass = D3D10_INPUT_PER_VERTEX_DATA;
	descEntry.InstanceDataStepRate = 0;
	_vertexDescr.push_back(descEntry);
	offset += 12;
	_vertexSize += 3;

	descEntry.SemanticName = "TANGENT";
	descEntry.SemanticIndex = 0;
	descEntry.Format = DXGI_FORMAT_R32G32B32_FLOAT;
	descEntry.InputSlot = 0;
	descEntry.AlignedByteOffset = offset;
	descEntry.InputSlotClass = D3D10_INPUT_PER_VERTEX_DATA;
	descEntry.InstanceDataStepRate = 0;
	_vertexDescr.push_back(descEntry);
	_vertexSize += 3;

	std::vector<float > bufferData;
	for(unsigned int i = 0; i < _numVertexs; i++)
	{
		bufferData.push_back(vertexs[i*3]);
		bufferData.push_back(vertexs[(i*3)+1]);
		bufferData.push_back(vertexs[(i*3)+2]);

		bufferData.push_back(normals[i*3]);
		bufferData.push_back(normals[(i*3)+1]);
		bufferData.push_back(normals[(i*3)+2]);

		bufferData.push_back(curv[i*3]);
		bufferData.push_back(curv[(i*3)+1]);
		bufferData.push_back(curv[(i*3)+2]);
	}

	if(_vertexBuffer)
	{
		_vertexBuffer->Release();
		_vertexBuffer = 0;
	}
	if(_indexBuffer)
	{
		_indexBuffer->Release();
		_indexBuffer = 0;
	}
	if(_adyIndexBuffer)
	{
		_adyIndexBuffer->Release();
		_adyIndexBuffer = 0;
	}

	//Create the vertex buffer
	D3D10_BUFFER_DESC vertexsDescr;
	vertexsDescr.Usage = D3D10_USAGE_IMMUTABLE;
	vertexsDescr.ByteWidth = sizeof(float)*_vertexSize*_numVertexs;
	vertexsDescr.BindFlags = D3D10_BIND_VERTEX_BUFFER;
	vertexsDescr.CPUAccessFlags = 0;
	vertexsDescr.MiscFlags = 0;
	D3D10_SUBRESOURCE_DATA vertexInitData;
	vertexInitData.pSysMem = &bufferData[0];
	_d3dDevice->CreateBuffer(&vertexsDescr,&vertexInitData,&_vertexBuffer);

	//Create the index buffer
	D3D10_BUFFER_DESC indexDescr;
    indexDescr.Usage           = D3D10_USAGE_IMMUTABLE;
    indexDescr.ByteWidth       = sizeof(unsigned int) * faces.size();
    indexDescr.BindFlags       = D3D10_BIND_INDEX_BUFFER;
    indexDescr.CPUAccessFlags  = 0;
    indexDescr.MiscFlags       = 0;
    D3D10_SUBRESOURCE_DATA indexInitData;
    indexInitData.pSysMem = &faces[0];
    indexInitData.SysMemPitch = 0;
    indexInitData.SysMemSlicePitch = 0;
	_d3dDevice->CreateBuffer(&indexDescr,&indexInitData,&_indexBuffer);

	//Create the adyacent index buffer
	D3D10_BUFFER_DESC adyIndexDescr;
    adyIndexDescr.Usage           = D3D10_USAGE_IMMUTABLE;
    adyIndexDescr.ByteWidth       = sizeof(unsigned int) * adyFaces.size();
    adyIndexDescr.BindFlags       = D3D10_BIND_INDEX_BUFFER;
    adyIndexDescr.CPUAccessFlags  = 0;
    adyIndexDescr.MiscFlags       = 0;
    D3D10_SUBRESOURCE_DATA adyIndexInitData;
    adyIndexInitData.pSysMem = &adyFaces[0];
    adyIndexInitData.SysMemPitch = 0;
    adyIndexInitData.SysMemSlicePitch = 0;
	_d3dDevice->CreateBuffer(&adyIndexDescr,&adyIndexInitData,&_adyIndexBuffer);

}