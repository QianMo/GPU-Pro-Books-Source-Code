#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

#include "../Level/LevelLoader.h"
#include "../XMLParser/XmlStream.h"
#include "../Render/RenderManager.h"
#include "../Render/RenderNode.h"
#include "../Render/RenderObject.h"
#include "../Render/Mesh.h"
#include "../Physic/Physic.h"

#include "../Util/Math.h"

// -----------------------------------------------------------------------------
// ----------------------- LevelLoader::LevelLoader ----------------------------
// -----------------------------------------------------------------------------
LevelLoader::LevelLoader(void)
{
	cameraPosition = Vector3::ZERO;
	cameraDirection = Vector3::Z_AXIS;
}


// -----------------------------------------------------------------------------
// ----------------------- LevelLoader::~LevelLoader ----------------------------
// -----------------------------------------------------------------------------
LevelLoader::~LevelLoader(void)
{
}


// -----------------------------------------------------------------------------
// ----------------------- LevelLoader::LoadLevel ------------------------------
// -----------------------------------------------------------------------------
void LevelLoader::LoadLevel(const char* fileName, bool createJoint)
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
		return;
	}

	string str = strm.str();

	currentNode = NODETYPE_NONE;
	currentGameElement = NULL;

	// parse document
	XmlStream xml;
	xml.setSubscriber(*this);
	xml.parse( (char *) str.c_str(), (long)str.size() - 1 );
	
	gameElements.push_back(currentGameElement);

	list<Physic::physicData*> physicDataList;

	createLevelTree(gameElements, physicDataList);

	Physic::Instance()->CreateLevel(physicDataList, createJoint);

	destroyGameElements(gameElements);
	destroyPhysicsData(physicDataList);

	gameElements.clear();
	physicDataList.clear();
}

// -----------------------------------------------------------------------------
// ----------------------- LevelLoader::foundNode ------------------------------
// -----------------------------------------------------------------------------
void LevelLoader::foundNode(string & name, string & attributes)
{
	if (name == "gameElement")
	{
		if (currentNode != NODETYPE_NONE)
			gameElements.push_back(currentGameElement);

		currentNode = NODETYPE_GAMEELEMENT;

		currentGameElement = new gameElement;
		currentGameElement->type = string(attributes.begin() + attributes.find("type=") + 5, attributes.begin() + attributes.find(" "));

		currentGameElement->position = Vector3(0.0f, 0.0f, 0.0f);
		currentGameElement->rotation = Vector3(0.0f, 0.0f, 0.0f);
		currentGameElement->meshFileName = "";
		currentGameElement->collFileNames.clear();
		currentGameElement->density = 1.0f;
		currentGameElement->measures = Vector3(1.0f, 1.0f, 1.0f);
		currentGameElement->uTex = Vector3(1.0f, 1.0f, 1.0f);
		currentGameElement->vTex = Vector3(1.0f, 1.0f, 1.0f);
		currentGameElement->material = -1;
	}
	else if (name == "elementPosition")
	{
		currentNode = NODETYPE_ELEMENTPOSITION;
	}
	else if (name == "elementRotation")
	{
		currentNode = NODETYPE_ELEMENTROTATION;
	}
	else if (name == "elementMeshFileName")
	{
		currentNode = NODETYPE_MESHFILENAME;
	}
	else if (name == "elementCollFileName")
	{
		currentNode = NODETYPE_COLLFILENAME;
	}
	else if (name == "elementMeasures")
	{
		currentNode = NODETYPE_ELEMENTMEASURES;
	}
	else if (name == "elementUTex")
	{
		currentNode = NODETYPE_ELEMENTUTEX;
	}
	else if (name == "elementVTex")
	{
		currentNode = NODETYPE_ELEMENTVTEX;
	}
	else if (name == "elementMaterial")
	{
		currentNode = NODETYPE_MATERIAL;
	}
	else if (name == "elementDensity")
	{
		currentNode = NODETYPE_DENSITY;
	}
	else if (name == "elementRadius")
	{
		currentNode = NODETYPE_RADIUS;
	}
	else if (name == "elementPrecision")
	{
		currentNode = NODETYPE_PRECISION;
	}
	else if (name == "elementTextureFileNames")
	{
		currentNode = NODETYPE_TEXTUREFILENAMES;
	}
	else if (name == "level")
	{
		//currentNode = NODETYPE_LEVEL;
	}
	else
	{
		assert(false);
	}
}


// -----------------------------------------------------------------------------
// ----------------------- LevelLoader::foundElement ---------------------------
// -----------------------------------------------------------------------------
void LevelLoader::foundElement(string & name, string & value, string & attributes)
{
	switch(currentNode)
	{
	case NODETYPE_NONE:
		break;
	case NODETYPE_LEVEL:
		break;
	case NODETYPE_GAMEELEMENT:
		break;
	case NODETYPE_ELEMENTPOSITION:
		if (name == "positionX")
		{
			currentGameElement->position.x = (float)atof(value.c_str());
		}
		else if (name == "positionY")
		{
			currentGameElement->position.y = (float)atof(value.c_str());
		}
		else if (name == "positionZ")
		{
			currentGameElement->position.z = (float)atof(value.c_str());
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_ELEMENTROTATION:
		if (name == "rotationX")
		{
			currentGameElement->rotation.x = ((float)atof(value.c_str()))*Math::DEG_TO_RAD;
		}
		else if (name == "rotationY")
		{
			currentGameElement->rotation.y = ((float)atof(value.c_str()))*Math::DEG_TO_RAD;
		}
		else if (name == "rotationZ")
		{
			currentGameElement->rotation.z = ((float)atof(value.c_str()))*Math::DEG_TO_RAD;
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_MESHFILENAME:
		if (name == "fileName")
		{
			currentGameElement->meshFileName = value;
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_COLLFILENAME:
		if (name == "fileName")
		{
			currentGameElement->collFileNames.push_back(value);
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_ELEMENTMEASURES:
		if (name == "width")
		{
			currentGameElement->measures.x = (float)atof(value.c_str());
		}
		else if (name == "height")
		{
			currentGameElement->measures.y = (float)atof(value.c_str());
		}
		else if (name == "depth")
		{
			currentGameElement->measures.z = (float)atof(value.c_str());
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_ELEMENTUTEX:
		if (name == "uTexX")
		{
			currentGameElement->uTex.x = (float)atof(value.c_str());
		}
		else if (name == "uTexY")
		{
			currentGameElement->uTex.y = (float)atof(value.c_str());
		}
		else if (name == "uTexZ")
		{
			currentGameElement->uTex.z = (float)atof(value.c_str());
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_ELEMENTVTEX:
		if (name == "vTexX")
		{
			currentGameElement->vTex.x = (float)atof(value.c_str());
		}
		else if (name == "vTexY")
		{
			currentGameElement->vTex.y = (float)atof(value.c_str());
		}
		else if (name == "vTexZ")
		{
			currentGameElement->vTex.z = (float)atof(value.c_str());
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_MATERIAL:
		if (name == "materialId")
		{
			currentGameElement->material = (int)atoi(value.c_str());
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_TEXTUREFILENAMES:
		if (name == "textureLeft")
		{
			skyBoxTextureFileNames[0] = value;
		}
		else if (name == "textureRight")
		{
			skyBoxTextureFileNames[1] = value;
		}
		else if (name == "textureTop")
		{
			skyBoxTextureFileNames[2] = value;
		}
		else if (name == "textureBottom")
		{
			skyBoxTextureFileNames[3] = value;
		}
		else if (name == "textureFront")
		{
			skyBoxTextureFileNames[4] = value;
		}
		else if (name == "textureBack")
		{
			skyBoxTextureFileNames[5] = value;
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_DENSITY:
		if (name == "density")
		{
			currentGameElement->density = (float)atof(value.c_str());
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_RADIUS:
		if (name == "radius")
		{
			currentGameElement->radius = (float)atof(value.c_str());
		}
		else
		{
			assert(false);
		}
		break;
	case NODETYPE_PRECISION:
		if (name == "precision")
		{
			currentGameElement->precision = (int)atoi(value.c_str());
		}
		else
		{
			assert(false);
		}
		break;
	}
}


// -----------------------------------------------------------------------------
// ----------------------- LevelLoader::startElement ---------------------------
// -----------------------------------------------------------------------------
void LevelLoader::startElement(string & name, string & value, string & attributes)
{
}


// -----------------------------------------------------------------------------
// ----------------------- LevelLoader::endElement -----------------------------
// -----------------------------------------------------------------------------
void LevelLoader::endElement(string & name, string & value, string & attributes)
{
}

// -----------------------------------------------------------------------------
// ----------------------- LevelLoader::createLevelTree ------------------------
// -----------------------------------------------------------------------------
void LevelLoader::createLevelTree(list<gameElement*>& geList, list<Physic::physicData*>& pChilds)
{
	list<gameElement*>::iterator i;
	for (i=geList.begin(); i!=geList.end(); ++i)
	{
		gameElement* element = *i;
		RenderObject* currentRenderObject = NULL;
		Physic::physicData* pData = NULL;
		unsigned int key = 0;

		if (element->type == "box")
		{
			{
				pData = new Physic::physicData;
				pData->type = element->type;

				key = RenderManager::Instance()->GetFreeKey();
				pData->matrixId = key;

				pData->position = element->position;
				pData->rotation = element->rotation;
				pData->measures = element->measures;
				pData->density  = element->density;

				pData->collMesh = false;

				pChilds.push_back(pData);
			}

			currentRenderObject = RenderManager::Instance()->CreateBox(element->position, element->rotation,
				element->measures.x, element->measures.y, element->measures.z,
				element->uTex, element->vTex, element->material, true, false);
		}
		else if ((element->type == "sphere") || (element->type == "dynamic_sphere"))
		{
			{
				pData = new Physic::physicData;
				pData->type = element->type;

				key = RenderManager::Instance()->GetFreeKey();
				pData->matrixId = key;

				pData->position = element->position;
				pData->rotation = element->rotation;
				pData->density  = element->density;
				pData->radius = element->radius;

				pData->collMesh = false;

				pChilds.push_back(pData);
			}

			if (element->meshFileName.compare("") != 0)
			{
				currentRenderObject = RenderManager::Instance()->CreateMesh(element->meshFileName.c_str(), element->position, element->rotation, element->material, true, false);
			}
			else
			{
				currentRenderObject = RenderManager::Instance()->CreateSphere(element->radius,
					element->precision,
					element->position,
					element->rotation,
					element->material, true, false);
			}
		}
		else if (element->type == "non_physics_mesh")
		{
			currentRenderObject = RenderManager::Instance()->CreateMesh(element->meshFileName.c_str(), element->position, element->rotation, element->material, false, false);
		}
		else if ((element->type == "mesh") || (element->type == "convex_mesh") || (element->type == "dynamic_mesh") || (element->type == "dynamic_convex_mesh"))
		{
			{
				pData = new Physic::physicData;
				pData->type = element->type;

				key = RenderManager::Instance()->GetFreeKey();
				pData->matrixId = key;

				pData->position = element->position;
				pData->rotation = element->rotation;
				pData->density  = element->density;

				if (element->collFileNames.size() > 0)
					pData->collMesh = true;
				else
					pData->collMesh = false;

				pChilds.push_back(pData);
			}

			if (element->collFileNames.size() > 0)
			{
				currentRenderObject = RenderManager::Instance()->CreateMesh(element->meshFileName.c_str(), element->position, element->rotation, element->material, true, false);

				unsigned int j;
				for (j=0; j<element->collFileNames.size(); j++)
				{
					Mesh* mesh = new Mesh();
					mesh->InitCollisionMesh(element->collFileNames[j].c_str());
					pData->objects.push_back(mesh);
				}
			}
			else
			{
				currentRenderObject = RenderManager::Instance()->CreateMesh(element->meshFileName.c_str(), element->position, element->rotation, element->material, true, false);
				pData->objects.push_back(currentRenderObject);
			}
		}
		else if (element->type == "skyBox")
		{
			const char** fileNames = new const char*[6];
			unsigned int i;
			for (i=0; i<6; i++)
			{
				fileNames[i] = skyBoxTextureFileNames[i].c_str();
			}
			RenderManager::Instance()->InitSkyBox(fileNames);

			delete fileNames;
		}
		else if (element->type == "cameraPosition")
		{
			cameraPosition = element->position;
		}
		else if (element->type == "cameraDirection")
		{
			cameraDirection = element->position;
		}
		else
		{
			assert(false);
		}

		if (currentRenderObject != NULL)
		{
			RenderNode* node = new RenderNode();
			node->SetMatrixId(key);
			node->AddElement(currentRenderObject);
			RenderManager::Instance()->AddNodeToRoot(node);
		}

		printf(".");
	}
}

// -----------------------------------------------------------------------------
// ----------------------- LevelLoader::destroyGameElements --------------------
// -----------------------------------------------------------------------------
void LevelLoader::destroyGameElements(list<gameElement*> gE)
{
	list<gameElement*>::iterator i;
	for (i = gE.begin(); i != gE.end(); ++i)
	{
		gameElement* object = *i;
		delete object;
	}
	gE.clear();
}

// -----------------------------------------------------------------------------
// ----------------------- LevelLoader::destroyGameElements --------------------
// -----------------------------------------------------------------------------
void LevelLoader::destroyPhysicsData(list<Physic::physicData*> pD)
{
	list<Physic::physicData*>::iterator i;
	for (i = pD.begin(); i != pD.end(); ++i)
	{
		Physic::physicData* object = *i;
		if (object->childs.size() > 0)
			destroyPhysicsData(object->childs);

		delete object;
	}
	pD.clear();
}