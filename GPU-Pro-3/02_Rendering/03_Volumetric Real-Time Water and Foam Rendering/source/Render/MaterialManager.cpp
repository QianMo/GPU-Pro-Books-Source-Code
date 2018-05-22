#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

#include <windows.h>
#include "GL/glew.h"
#include <GL/gl.h>

#include "../Render/MaterialManager.h"
#include "../Graphics/TextureManager.h"
#include "../Render/ShaderManager.h"

#include "../XMLParser/XmlStream.h"

#include <GL/glut.h>

// -----------------------------------------------------------------------------
// ------------------- MaterialManager::MaterialManager ------------------------
// -----------------------------------------------------------------------------
MaterialManager::MaterialManager(void):
	isInit(false)
{
}

// -----------------------------------------------------------------------------
// ------------------- MaterialManager::~MaterialManager -----------------------
// -----------------------------------------------------------------------------
MaterialManager::~MaterialManager(void)
{
	Exit();
}

// -----------------------------------------------------------------------------
// ------------------- MaterialManager::LoadMaterialsFromFile ------------------
// -----------------------------------------------------------------------------
void MaterialManager::LoadMaterialsFromFile(const char* fileName)
{
	stringstream strm;
	string line;

	// open xml file
	std::ifstream file(fileName);
	if(file) {
		// loop until end of file
		while(!file.eof()) {
			getline(file, line);
			strm << line << endl;
		}
		// close xml file
		file.close();
	}
	else
	{
		assert(false);
	}

	string str = strm.str();

	currentNode = NODETYPE_NONE;

	// parse document
	XmlStream xml;
	xml.setSubscriber(*this);
	xml.parse( (char *) str.c_str(), (long)str.size() - 1 );

	// the last material has to be added too...
	AddCurrentMaterial();

	isInit = true;
}

// -----------------------------------------------------------------------------
// ------------------- MaterialManager::AddMaterial ----------------------------
// -----------------------------------------------------------------------------
const int MaterialManager::AddMaterial(const Material& mat)
{
	assert((int)materials.size() <= 32);

	int key = (int)materials.size()+1;
	materials.insert(std::make_pair(key, mat));

	return key;
}

// -----------------------------------------------------------------------------
// ------------------- MaterialManager::SetMaterial ----------------------------
// -----------------------------------------------------------------------------
void MaterialManager::SetMaterial(const int& key)
{
	assert(isInit == true);

	if (key == 0)
		return;

	iter = materials.find(key);
	assert((int)materials.count(key) > 0);
	{
		Material mat = iter->second;
		Material::MaterialDefinition def = mat.GetMaterial();

		if (def.lighting)
		{
			ShaderManager::Instance()->SetParameter3fv(ShaderManager::SP_KA, def.ambiente);
			ShaderManager::Instance()->SetParameter4fv(ShaderManager::SP_KD, def.diffuse);
			ShaderManager::Instance()->SetParameter3fv(ShaderManager::SP_KS, def.specular);
			ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SHININESS, def.shininess);
		}

		if (def.textureId != 0)
		{
			ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_TEXTURE, def.textureId);
		}
		if (def.normalMapId != 0)
		{
			ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_NORMAL_MAP, def.normalMapId);
		}
	}
}

// -----------------------------------------------------------------------------
// ------------------- MaterialManager::GetMaterial ----------------------------
// -----------------------------------------------------------------------------
Material::MaterialDefinition MaterialManager::GetMaterial(const int& key)
{
	iter = materials.find(key);
	assert((int)materials.count(key) > 0);
	
	Material mat = iter->second;
	return mat.GetMaterial();
}

// -----------------------------------------------------------------------------
// ------------------- MaterialManager::CheckMaterial --------------------------
// -----------------------------------------------------------------------------
bool MaterialManager::CheckMaterial(const int& key)
{
	assert(isInit == true);

	iter = materials.find(key);
	assert((int)materials.count(key) > 0);
	if((int)materials.count(key) > 0)
		return true;
	else
		return false;
}

// -----------------------------------------------------------------------------
// ------------------- MaterialManager::Exit -----------------------------------
// -----------------------------------------------------------------------------
void MaterialManager::Exit(void)
{
	materials.clear();
}

// -----------------------------------------------------------------------------
// ------------------- MaterialManager::foundNode ------------------------------
// -----------------------------------------------------------------------------
void MaterialManager::foundNode(string & name, string & attributes)
{
	if (name == "material")
	{
		if (currentNode != NODETYPE_NONE)
		{
			AddCurrentMaterial();
		}
		currentNode = NODETYPE_MATERIAL;

		int i;
		for (i=0; i<4; i++)
		{
			currentMaterial.ambiente[i] = 0.0f;
			currentMaterial.diffuse[i] = 0.0f;
			currentMaterial.specular[i] = 0.0f;
		}
		currentMaterial.shininess = 0.0f;
		currentMaterial.texId = 0;
		currentMaterial.normalMapId = 0;
		currentMaterial.useParallaxMapping = false;
	}
	else if (name == "lighting")
	{
		currentNode = NODETYPE_LIGHTING;
	}
	else if (name == "ambiente")
	{
		currentNode = NODETYPE_AMBIENTE;
	}
	else if (name == "diffuse")
	{
		currentNode = NODETYPE_DIFFUSE;
	}
	else if (name == "specular")
	{
		currentNode = NODETYPE_SPECULAR;
	}
	else if (name == "shininess")
	{
		currentNode = NODETYPE_SHININESS;
	}
	else if (name == "texture")
	{
		currentNode = NODETYPE_TEXTUREFILENAME;
	}
	else if (name == "normalMap")
	{
		currentNode = NODETYPE_NORMALMAPFILENAME;
	}
	else
	{
		assert(false);
	}
}


// -----------------------------------------------------------------------------
// ------------------- MaterialManager::foundElement ---------------------------
// -----------------------------------------------------------------------------
void MaterialManager::foundElement(string & name, string & value, string & attributes)
{
	switch(currentNode)
	{
	case NODETYPE_NONE:
		break;
	case NODETYPE_MATERIAL:
		break;
	case NODETYPE_LIGHTING:
		if (name == "useLight")
		{
			if (value == "true")
				currentMaterial.lighting = true;
			else if (value == "false")
				currentMaterial.lighting = false;
			else
				assert(false);
		}
		break;
	case NODETYPE_AMBIENTE:
		if (name == "red")
		{
			currentMaterial.ambiente[0] = (float)atof(value.c_str());
		}
		else if (name == "green")
		{
			currentMaterial.ambiente[1] = (float)atof(value.c_str());
		}
		else if (name == "blue")
		{
			currentMaterial.ambiente[2] = (float)atof(value.c_str());
		}
		else if (name == "alpha")
		{
			currentMaterial.ambiente[3] = (float)atof(value.c_str());
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_DIFFUSE:
		if (name == "red")
		{
			currentMaterial.diffuse[0] = (float)atof(value.c_str());
		}
		else if (name == "green")
		{
			currentMaterial.diffuse[1] = (float)atof(value.c_str());
		}
		else if (name == "blue")
		{
			currentMaterial.diffuse[2] = (float)atof(value.c_str());
		}
		else if (name == "alpha")
		{
			currentMaterial.diffuse[3] = (float)atof(value.c_str());
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_SPECULAR:
		if (name == "red")
		{
			currentMaterial.specular[0] = (float)atof(value.c_str());
		}
		else if (name == "green")
		{
			currentMaterial.specular[1] = (float)atof(value.c_str());
		}
		else if (name == "blue")
		{
			currentMaterial.specular[2] = (float)atof(value.c_str());
		}
		else if (name == "alpha")
		{
			currentMaterial.specular[3] = (float)atof(value.c_str());
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_SHININESS:
		if (name == "shininessValue")
		{
			currentMaterial.shininess = (float)atof(value.c_str());
		}
		else
		{
			assert(false);
		}
		break;
	
	case NODETYPE_TEXTUREFILENAME:
		if (name == "textureFileName")
		{
			if (value != "")
				currentMaterial.texId = TextureManager::Instance()->LoadTexture(value.c_str());
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_NORMALMAPFILENAME:
		if (name == "normalMapFileName")
		{
			if (value != "")
				currentMaterial.normalMapId = TextureManager::Instance()->LoadTexture(value.c_str());
			currentMaterial.useParallaxMapping = true;
		}
		else
		{
			assert(false);
		}
		break;
	default:
		assert(false);
		break;
	}
}


// -----------------------------------------------------------------------------
// ----------------------- MaterialManager::startElement -----------------------
// -----------------------------------------------------------------------------
void MaterialManager::startElement(string & name, string & value, string & attributes)
{
}


// -----------------------------------------------------------------------------
// ----------------------- MaterialManager::endElement -------------------------
// -----------------------------------------------------------------------------
void MaterialManager::endElement(string & name, string & value, string & attributes)
{
}

// -----------------------------------------------------------------------------
// --------------- MaterialManager::AddCurrentMaterial -------------------------
// -----------------------------------------------------------------------------
void MaterialManager::AddCurrentMaterial(void)
{
	Material::MaterialDefinition matDef;

	matDef.lighting = currentMaterial.lighting;

	int i;
	for (i=0; i<4; i++)
	{
		matDef.ambiente[i] = currentMaterial.ambiente[i];
		matDef.diffuse[i] = currentMaterial.diffuse[i];
		matDef.specular[i] = currentMaterial.specular[i];
	}
	matDef.shininess = currentMaterial.shininess;
	matDef.textureId = currentMaterial.texId;
	matDef.normalMapId = currentMaterial.normalMapId;
	matDef.useParallaxMapping = currentMaterial.useParallaxMapping;

	Material mat;
	mat.Init(matDef);

	AddMaterial(mat);
}