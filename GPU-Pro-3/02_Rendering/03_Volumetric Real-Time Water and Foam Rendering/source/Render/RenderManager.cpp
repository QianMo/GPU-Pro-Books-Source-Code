#include "GL/glew.h"

#include <assert.h>

#include "../Render/RenderManager.h"
#include "../Render/MaterialManager.h"
#include "../Render/RenderNode.h"
#include "../Render/ShaderManager.h"
#include "../Render/Sphere.h"
#include "../Render/Box.h"
#include "../Render/Mesh.h"

#include "../Graphics/TextureManager.h"

#include "../Util/Vector3.h"

#include "../Physic/Physic.h"

#include <GL/glut.h>


// -----------------------------------------------------------------------------
// ----------------------- RenderManager::RenderManager ------------------------
// -----------------------------------------------------------------------------
RenderManager::RenderManager(void) :
	root(NULL)
{
}

// -----------------------------------------------------------------------------
// ----------------------- RenderManager::~RenderManager -----------------------
// -----------------------------------------------------------------------------
RenderManager::~RenderManager(void)
{
	Exit();
}

// -----------------------------------------------------------------------------
// ----------------------- RenderManager::Init ---------------------------------
// -----------------------------------------------------------------------------
void RenderManager::Init(void)
{
	keyCounter = 1;
	uniqueCounter = 0;

	root = new RenderNode();
}

// -----------------------------------------------------------------------------
// ----------------------- RenderManager::RenderManager ------------------------
// -----------------------------------------------------------------------------
void RenderManager::InitSkyBox(const char** fileNames)
{
	unsigned int i;
	for (i=0; i<6; i++)
	{
		skyBoxTextures[i] = TextureManager::Instance()->LoadTexture(fileNames[i], false, true, true);
	}
}

// -----------------------------------------------------------------------------
// ----------------------- RenderManager::Update --------------------------------
// -----------------------------------------------------------------------------
void RenderManager::Update(float deltaTime)
{
	UpdateTreeNode(root, deltaTime, Matrix4::IDENTITY);
}

// -----------------------------------------------------------------------------
// ----------------------- RenderManager::Render -------------------------------
// -----------------------------------------------------------------------------
void RenderManager::Render(void)
{
	if (skipMaterials)
		MaterialManager::Instance()->SetMaterial(1);

	RenderTreeNode(root);
}

// -----------------------------------------------------------------------------
// ----------------------- RenderManager::RenderSkyBox -------------------------
// -----------------------------------------------------------------------------
void RenderManager::RenderSkyBox(Camera* camera)
{
	glDisable(GL_DEPTH_TEST);
	glColor3f(1, 1, 1);

	float dim = 20.0f;

	glPushMatrix();
	{
		camera->SetViewMatrixCentered();

		// left
		ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_SKYBOX_FACE, skyBoxTextures[0]);
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SKYBOX, "Skybox");

		glBegin(GL_QUADS);		
		glTexCoord2f(0.0f, 0.0f); glVertex3f(-dim, dim, -dim);	
		glTexCoord2f(1.0f, 0.0f); glVertex3f(-dim, dim, dim); 
		glTexCoord2f(1.0f, 1.0f); glVertex3f(-dim, -dim, dim);
		glTexCoord2f(0.0f, 1.0f); glVertex3f(-dim, -dim, -dim);		
		glEnd();

		ShaderManager::Instance()->DisableShader();

		// right
		ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_SKYBOX_FACE, skyBoxTextures[1]);
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SKYBOX, "Skybox");

		glBegin(GL_QUADS);		
		glTexCoord2f(1.0f, 1.0f); glVertex3f(dim, -dim, -dim);
		glTexCoord2f(0.0f, 1.0f); glVertex3f(dim, -dim, dim);
		glTexCoord2f(0.0f, 0.0f); glVertex3f(dim, dim, dim); 
		glTexCoord2f(1.0f, 0.0f); glVertex3f(dim, dim, -dim);
		glEnd();

		ShaderManager::Instance()->DisableShader();

		// top
		ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_SKYBOX_FACE, skyBoxTextures[2]);
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SKYBOX, "Skybox");

		glBegin(GL_QUADS);		
		glTexCoord2f(0.0f, 1.0f); glVertex3f(dim, dim, -dim);
		glTexCoord2f(0.0f, 0.0f); glVertex3f(dim, dim, dim); 
		glTexCoord2f(1.0f, 0.0f); glVertex3f(-dim,	dim, dim);
		glTexCoord2f(1.0f, 1.0f); glVertex3f(-dim, dim, -dim);
		glEnd();

		ShaderManager::Instance()->DisableShader();

		// bottom
		ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_SKYBOX_FACE, skyBoxTextures[3]);
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SKYBOX, "Skybox");

		glBegin(GL_QUADS);		
		glTexCoord2f(1.0f, 0.0f); glVertex3f(-dim, -dim, -dim);
		glTexCoord2f(1.0f, 1.0f); glVertex3f(-dim, -dim, dim);
		glTexCoord2f(0.0f, 1.0f); glVertex3f(dim, -dim, dim); 
		glTexCoord2f(0.0f, 0.0f); glVertex3f(dim, -dim, -dim);
		glEnd();

		ShaderManager::Instance()->DisableShader();

		// front
		ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_SKYBOX_FACE, skyBoxTextures[4]);
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SKYBOX, "Skybox");

		glBegin(GL_QUADS);	
		glTexCoord2f(0.0f, 1.0f); glVertex3f(-dim, -dim, dim);
		glTexCoord2f(0.0f, 0.0f); glVertex3f(-dim,	dim, dim);
		glTexCoord2f(1.0f, 0.0f); glVertex3f(dim, dim, dim); 
		glTexCoord2f(1.0f, 1.0f); glVertex3f(dim, -dim, dim);
		glEnd();

		ShaderManager::Instance()->DisableShader();

		// back
		ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_SKYBOX_FACE, skyBoxTextures[5]);
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SKYBOX, "Skybox");

		glBegin(GL_QUADS);		
		glTexCoord2f(0.0f, 1.0f); glVertex3f(dim, -dim, -dim);
		glTexCoord2f(0.0f, 0.0f); glVertex3f(dim, dim, -dim); 
		glTexCoord2f(1.0f, 0.0f); glVertex3f(-dim, dim, -dim);
		glTexCoord2f(1.0f, 1.0f); glVertex3f(-dim, -dim, -dim);
		glEnd();

		ShaderManager::Instance()->DisableShader();
	}
	glPopMatrix();

	glEnable(GL_DEPTH_TEST);
}


// -----------------------------------------------------------------------------
// ----------------------- RenderManager::Exit ---------------------------------
// -----------------------------------------------------------------------------
void RenderManager::Exit(void)
{
	if (root)
	{
		DestroyRenderTree(root);
		renderObjects.clear();
		delete root;
		root = NULL;
	}
}

// -----------------------------------------------------------------------------
// ------------------ RenderManager::AddNodeToRoot -----------------------------
// -----------------------------------------------------------------------------
void RenderManager::AddNodeToRoot(RenderNode* node)
{
	assert(root != NULL);
	root->AddChild(node);
	node->SetParent(root);
}

// -----------------------------------------------------------------------------
// ----------------- RenderManager::AddElementToRoot ---------------------------
// -----------------------------------------------------------------------------
void RenderManager::AddElementToRoot(RenderObject* element)
{
	assert(root != NULL);
	root->AddElement(element);
}

// -----------------------------------------------------------------------------
// ----------------- RenderManager::AddElementToRoot ---------------------------
// -----------------------------------------------------------------------------
void RenderManager::RemoveNodeFromRoot(const char* nodeName)
{
	list<RenderNode*>::iterator childIter;
	for (childIter = root->GetChildBegin(); childIter != root->GetChildEnd(); ++childIter)
	{
		RenderNode* nextNode = *childIter;
		if (strcmp(nextNode->GetNodeName(), nodeName) == 0)
		{
			DestroyRenderTree(nextNode);
		}
	}
}


// -----------------------------------------------------------------------------
// ----------------------- RenderManager::SetMatrix ----------------------------
// -----------------------------------------------------------------------------
void RenderManager::SetMatrix(const unsigned int& key, const Matrix4& world)
{
	if (renderObjects.empty())
		return;

	assert(renderObjects.count(key) > 0);
	iter = renderObjects.find(key);
	RenderObject* obj = iter->second;
	obj->SetMatrix(world);
}

// -----------------------------------------------------------------------------
// ----------------------- RenderManager::AddSphere ----------------------------
// -----------------------------------------------------------------------------
void RenderManager::AddSphere(unsigned int& uniqueId, unsigned int& key, const float& r, const int& p, const Vector3& pos, const int& matId, const bool& addToPhysicWorld, const bool& isLevelElement)
{
	char nodeName[64];

	Sphere* sphere = new Sphere();
	sphere->Init(uniqueCounter, r, p, pos, matId, addToPhysicWorld, isLevelElement);

	renderObjects.insert(std::make_pair(uniqueCounter, sphere));

	RenderNode* newNode = new RenderNode();
	if (addToPhysicWorld || isLevelElement)
	{
		newNode->SetMatrixId(keyCounter);
	}
	newNode->AddElement(sphere);

	sprintf_s(nodeName, "DynamicElement%.4d", keyCounter);
	newNode->SetNodeName(nodeName);
	
	AddNodeToRoot(newNode);

	if (addToPhysicWorld)
		Physic::Instance()->CreateSphere(keyCounter, NxVec3(pos.x, pos.y, pos.z), r);

	uniqueId = uniqueCounter;
	key = keyCounter;

	keyCounter++;
	uniqueCounter++;
}


// -----------------------------------------------------------------------------
// ----------------------- RenderManager::AddBox -------------------------------
// -----------------------------------------------------------------------------
const unsigned int RenderManager::AddBox(unsigned int& uniqueId, unsigned int& key, const Vector3& pos, const float& w, const float& h, const float& d, const Vector3& su, const Vector3& sv, const int& matId, const bool& addToPhysicWorld, const bool& isLevelElement)
{
	char nodeName[32];

	Box* box = new Box();
	box->Init(uniqueCounter, pos, w, h, d, su, sv, matId, addToPhysicWorld, isLevelElement);

	renderObjects.insert(std::make_pair(uniqueCounter, box));

	RenderNode* newNode = new RenderNode();
	if (addToPhysicWorld || isLevelElement)
	{
		newNode->SetMatrixId(keyCounter);
	}
	newNode->AddElement(box);

	sprintf_s(nodeName, "DynamicElement%.4d", keyCounter);
	newNode->SetNodeName(nodeName);

	AddNodeToRoot(newNode);

	if (addToPhysicWorld)
		Physic::Instance()->CreateBox(keyCounter, NxVec3(pos.x, pos.y, pos.z), NxVec3(w*0.5f, h*0.5f, d*0.5f));

	uniqueId = uniqueCounter;
	key = keyCounter;

	keyCounter++;
	uniqueCounter++;

	return uniqueCounter - 1;
}

// -----------------------------------------------------------------------------
// ----------------------- RenderManager::AddMesh ------------------------------
// -----------------------------------------------------------------------------
void RenderManager::AddMesh(unsigned int& uniqueId, unsigned int& key, const char* fileName, const int& matId, const bool& addToPhysicWorld, const bool& isLevelElement)
{
	Mesh* mesh = new Mesh();
	mesh->Init(uniqueCounter, fileName, matId, addToPhysicWorld, isLevelElement);

	renderObjects.insert(std::make_pair(uniqueCounter, mesh));

	RenderNode* newNode = new RenderNode();
	if (addToPhysicWorld || isLevelElement)
	{
		newNode->SetMatrixId(keyCounter);
	}
	newNode->AddElement(mesh);
	AddNodeToRoot(newNode);

	uniqueId = uniqueCounter;
	key = keyCounter;

	keyCounter++;
	uniqueCounter++;
}

// -----------------------------------------------------------------------------
// ----------------------- RenderManager::AddSphereWithKey ---------------------
// -----------------------------------------------------------------------------
const unsigned int RenderManager::AddSphereWithKey(const unsigned int& key, const float& r, const int& p, const Vector3& pos, const int& matId, const bool& addToPhysicWorld, const bool& isLevelElement)
{
	Sphere* sphere = new Sphere();
	sphere->Init(uniqueCounter, r, p, Vector3(0.0f, 0.0f, 0.0f), matId, addToPhysicWorld, isLevelElement);

	Matrix4 translation = Matrix4::Matrix4Translation(pos);
	sphere->SetMatrix(translation);

	renderObjects.insert(std::make_pair(uniqueCounter, sphere));

	RenderNode* newNode = new RenderNode();
	if (addToPhysicWorld || isLevelElement)
	{
		newNode->SetMatrixId(key);
	}
	newNode->AddElement(sphere);
	AddNodeToRoot(newNode);

	uniqueCounter++;

	return uniqueCounter - 1;
}


// -----------------------------------------------------------------------------
// ----------------------- RenderManager::AddBoxWithKey ------------------------
// -----------------------------------------------------------------------------
const unsigned int RenderManager::AddBoxWithKey(const unsigned int& key, const Vector3& pos, const float& w, const float& h, const float& d, const Vector3& su, const Vector3& sv, const int& matId, const bool& addToPhysicWorld, const bool& isLevelElement)
{
	Box* box = new Box();
	box->Init(uniqueCounter, Vector3(0.0f, 0.0f, 0.0f), w, h, d, su, sv, matId, addToPhysicWorld, isLevelElement);

	Matrix4 translation = Matrix4::Matrix4Translation(pos);
	box->SetMatrix(translation);

	renderObjects.insert(std::make_pair(uniqueCounter, box));

	RenderNode* newNode = new RenderNode();
	if (addToPhysicWorld || isLevelElement)
	{
		newNode->SetMatrixId(key);
	}
	newNode->AddElement(box);
	AddNodeToRoot(newNode);

	uniqueCounter++;

	return uniqueCounter - 1;
}

// -----------------------------------------------------------------------------
// ----------------------- RenderManager::AddMeshWithKey -----------------------
// -----------------------------------------------------------------------------
const unsigned int RenderManager::AddMeshWithKey(const unsigned int& key, const char* fileName, const Vector3& pos, const int& matId, const bool& addToPhysicWorld, const bool& isLevelElement)
{
	Mesh* mesh = new Mesh();
	mesh->Init(uniqueCounter, fileName, matId, addToPhysicWorld, isLevelElement);

	Matrix4 translation = Matrix4::Matrix4Translation(pos);
	mesh->SetMatrix(translation);

	renderObjects.insert(std::make_pair(uniqueCounter, mesh));

	RenderNode* newNode = new RenderNode();
	if (addToPhysicWorld || isLevelElement)
	{
		newNode->SetMatrixId(key);
	}
	newNode->AddElement(mesh);
	AddNodeToRoot(newNode);

	uniqueCounter++;

	return uniqueCounter - 1;
}

// -----------------------------------------------------------------------------
// ----------------------- RenderManager::CreateSphere -------------------------
// -----------------------------------------------------------------------------
RenderObject* RenderManager::CreateSphere(const float& r, const int& p, const Vector3& pos, const Vector3& rot, const int& matId, const bool& addToPhysicWorld, const bool& isLevelElement)
{
	Sphere* sphere = new Sphere();
	sphere->Init(uniqueCounter, r, p, Vector3(0.0f, 0.0f, 0.0f), matId, addToPhysicWorld, isLevelElement);

	if (!addToPhysicWorld)
	{
		Matrix4 transformation = Matrix4::Matrix4Rotation((rot.x), (rot.y), (rot.z)) * Matrix4::Matrix4Translation(pos);
		sphere->SetMatrix(transformation);
	}

	renderObjects.insert(std::make_pair(uniqueCounter, sphere));

	uniqueCounter++;

	return sphere;
}


// -----------------------------------------------------------------------------
// ----------------------- RenderManager::CreateBox ----------------------------
// -----------------------------------------------------------------------------
RenderObject* RenderManager::CreateBox(const Vector3& pos, const Vector3& rot, const float& w, const float& h, const float& d, const Vector3& su, const Vector3& sv, const int& matId, const bool& addToPhysicWorld, const bool& isLevelElement)
{
	Box* box = new Box();
	box->Init(uniqueCounter, Vector3(0.0f, 0.0f, 0.0f), w, h, d, su, sv, matId, addToPhysicWorld, isLevelElement);

	if (!addToPhysicWorld)
	{
		Matrix4 transformation = Matrix4::Matrix4Rotation((rot.x), (rot.y), (rot.z)) * Matrix4::Matrix4Translation(pos);
		box->SetMatrix(transformation);
	}

	renderObjects.insert(std::make_pair(uniqueCounter, box));

	uniqueCounter++;

	return box;
}

// -----------------------------------------------------------------------------
// ----------------------- RenderManager::CreateMesh ---------------------------
// -----------------------------------------------------------------------------
RenderObject* RenderManager::CreateMesh(const char* fileName, const Vector3& pos, const Vector3& rot, const int& matId, const bool& addToPhysicWorld, const bool& isLevelElement)
{
	Mesh* mesh = new Mesh();
	mesh->Init(uniqueCounter, fileName, matId, addToPhysicWorld, isLevelElement);

	if (!addToPhysicWorld)
	{
		Matrix4 transformation = Matrix4::Matrix4Rotation((rot.x), (rot.y), (rot.z)) * Matrix4::Matrix4Translation(pos);
		mesh->SetMatrix(transformation);
	}

	renderObjects.insert(std::make_pair(uniqueCounter, mesh));

	uniqueCounter++;

	return mesh;
}

// -----------------------------------------------------------------------------
// ----------------------- RenderManager::GetFreeKey ---------------------------
// -----------------------------------------------------------------------------
const unsigned int RenderManager::GetFreeKey(void)
{
	keyCounter++;
	return keyCounter - 1;
}

// -----------------------------------------------------------------------------
// ----------------- RenderManager::UpdateTreeNode -----------------------------
// -----------------------------------------------------------------------------
void RenderManager::UpdateTreeNode(RenderNode* node, float deltaTime, const Matrix4& matrix)
{
	node->Update(deltaTime, matrix);

	list<RenderNode*>::iterator childIter;
	for (childIter = node->GetChildBegin(); childIter != node->GetChildEnd(); ++childIter)
	{
		RenderNode* nextNode = *childIter;
		UpdateTreeNode(nextNode, deltaTime, matrix*nextNode->GetMatrix());
	}
}

// -----------------------------------------------------------------------------
// ----------------- RenderManager::RenderTreeNode -----------------------------
// -----------------------------------------------------------------------------
void RenderManager::RenderTreeNode(RenderNode* node)
{
	glPushMatrix();
	{
		node->RenderElements();

		list<RenderNode*>::iterator childIter;
		for (childIter = node->GetChildBegin(); childIter != node->GetChildEnd(); ++childIter)
		{
			RenderNode* nextNode = *childIter;
			RenderTreeNode(nextNode);
		}
	}
	glPopMatrix();
}

// -----------------------------------------------------------------------------
// ----------------- RenderManager::RenderTreeNode -----------------------------
// -----------------------------------------------------------------------------
void RenderManager::DestroyRenderTree(RenderNode* node)
{
	list<RenderNode*>::iterator childIter;
	for (childIter = node->GetChildBegin(); childIter != node->GetChildEnd(); ++childIter)
	{
		RenderNode* nextNode = *childIter;
		DestroyRenderTree(nextNode);
		delete nextNode;
	}

	node->DestroyChildList();
	node->Destroy();
}