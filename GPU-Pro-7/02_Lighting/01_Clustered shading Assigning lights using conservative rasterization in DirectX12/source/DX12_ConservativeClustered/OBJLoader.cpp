#include "OBJLoader.h"
#include <fstream>
#include <sstream>
#include <string>
#include "SharedContext.h"
#include "Log.h"

using namespace Log;

struct MatMap
{
	char mat[64];
	char tex[64];
};

OBJLoader::OBJLoader()
{

}

OBJLoader::~OBJLoader()
{

}

std::unique_ptr<Model> OBJLoader::LoadBIN(const char* file, ID3D12GraphicsCommandList* gfx_command_list)
{
	VertexIndexData data;
	std::vector<MeshGroup> meshGroups;
	std::vector<MatMap> matMap;
	std::ifstream binFile;

	binFile.open(file, std::ifstream::in | std::ifstream::binary);

	if(binFile.is_open())
	{
		uint32 vertexCount = 0;
		uint32 indexCount = 0;

		binFile.read((char*)&vertexCount, sizeof(uint32));
		binFile.read((char*)&indexCount, sizeof(uint32));

		data.vertexData.resize(vertexCount);
		data.indexData.resize(indexCount);

		binFile.read((char*)&data.vertexData[0], vertexCount * sizeof(Vert));
		binFile.read((char*)&data.indexData[0], indexCount * sizeof(uint32));

		uint32 groupCount = 0;

		binFile.read((char*)&groupCount, sizeof(uint32));

		meshGroups.resize(groupCount);

		binFile.read((char*)&meshGroups[0], groupCount * sizeof(MeshGroup));

		//read materials
		uint32 materialCount = 0;
		binFile.read((char*)&materialCount, sizeof(uint32));

		matMap.resize(materialCount);

		binFile.read((char*)&matMap[0], materialCount * sizeof(MatMap));

		binFile.close();
	}
	

	std::map<std::string, Material> materialMap;
	for(auto mm : matMap)
	{
		std::string textureName(mm.tex);
		std::string filePath = "../assets/models/" + textureName;

		Material mat;
		if(textureName == "")
		{
			mat.diffuse = nullptr;
		}
		else
		{
			mat.diffuse = new Texture();
			mat.diffuse->CreateTextureFromFile(gfx_command_list, filePath.c_str());
		}
		
		materialMap[mm.mat] = mat;
	}
	
	return  std::make_unique<Model>(&data, meshGroups, materialMap, gfx_command_list);
}

std::unique_ptr<LightShape> OBJLoader::LoadLightShape(const char* file, ID3D12GraphicsCommandList* gfx_command_list)
{
	std::fstream objFile;
	objFile.open(file, std::fstream::in);

	std::vector<Vector3> positions;

	VertexPIndexData data;

	if (objFile)
	{
		std::string line;
		std::string prefix;

		while (objFile.eof() == false)
		{
			prefix = "NULL";
			std::stringstream lineStream;

			getline(objFile, line);
			lineStream << line;
			lineStream >> prefix;


			if (prefix == "v")
			{
				Vector3 pos;
				lineStream >> pos.x >> pos.y >> pos.z;
				data.vertexData.push_back(VertP8(pos));
			}
			else if (prefix == "f")
			{
				uint16 indexPos;

				//Loop through vertex points in face
				for (unsigned i = 0; i < 3; ++i)
				{
					lineStream >> indexPos;

					data.indexData.push_back(indexPos - 1);
				}
			}
		}
	}
	PRINT(LogLevel::DEBUG_PRINT, "Light shape %s has loaded with %d vertices and %d indices!", file, data.vertexData.size(), data.indexData.size());
	return std::make_unique<LightShape>(&data, gfx_command_list);
}
