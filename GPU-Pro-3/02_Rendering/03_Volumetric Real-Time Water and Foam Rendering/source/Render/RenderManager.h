#ifndef __RENDERMANAGER__H__
#define __RENDERMANAGER__H__

#include <map>
#include <list>
#include <iostream>
#include <string>

#include "../Util/Singleton.h"
#include "../Util/Ray.h"

#include "../Render/RenderNode.h"

#include "../Level/Camera.h"

using std::map;
using std::string;
using std::list;

class RenderObject;
class Vector3;
class Matrix4;

class RenderManager : public Singleton<RenderManager>
{
	friend class Singleton<RenderManager>;

public:
	RenderManager(void);
	~RenderManager(void);

	// inits the render manager
	void Init(void);

	// inits the skybox
	void InitSkyBox(const char** fileNames);
	
	// updates the render objects
	void Update(float deltaTime);

	// renders all objects
	void Render(void);

	// renders the skybox
	void RenderSkyBox(Camera* camera);
	
	// exit the render manager
	void Exit(void);

	// adds a child to the root node
	void AddNodeToRoot(RenderNode* node);

	// adds a renderable element to the root node
	void AddElementToRoot(RenderObject* element);

	// removes a tree node (inluding his childs) from the root of the render tree
	void RemoveNodeFromRoot(const char* nodeName);

	// sets a matrix for a specified object in the world
	void SetMatrix(const unsigned int& key, const Matrix4& world);

	// adds a sphere to the rendering system (r: radius, p: precision)
	void AddSphere(unsigned int& uniqueId, unsigned int& key, const float& r, const int& p, const Vector3& pos, const int& matId=0, const bool& addToPhysicWorld=true, const bool& isLevelElement=false);
	
	// adds a box to the rendering system
	const unsigned int AddBox(unsigned int& uniqueId, unsigned int& key, const Vector3& pos, const float& w, const float& h, const float& d, const Vector3& su, const Vector3& sv, const int& matId=0, const bool& addToPhysicWorld=true, const bool& isLevelElement=false);

	// adds a mesh to the rendering system
	void AddMesh(unsigned int& uniqueId, unsigned int& key, const char* fileName, const int& matId=0, const bool& addToPhysicWorld=true, const bool& isLevelElement=false);
	
	// adds a sphere to the rendering system
	const unsigned int AddSphereWithKey(const unsigned int& key, const float& r, const int& p, const Vector3& pos, const int& matId=0, const bool& addToPhysicWorld=true, const bool& isLevelElement=false);

	// adds a box to the rendering system
	const unsigned int AddBoxWithKey(const unsigned int& key, const Vector3& pos, const float& w, const float& h, const float& d, const Vector3& su, const Vector3& sv, const int& matId=0, const bool& addToPhysicWorld=true, const bool& isLevelElement=false);

	// adds a mesh to the rendering system
	const unsigned int AddMeshWithKey(const unsigned int& key, const char* fileName, const Vector3& pos, const int& matId=0, const bool& addToPhysicWorld=true, const bool& isLevelElement=false);

	// creates a sphere
	RenderObject* CreateSphere(const float& r, const int& p, const Vector3& pos, const Vector3& rot, const int& matId=0, const bool& addToPhysicWorld=true, const bool& isLevelElement=false);

	// creates a box
	RenderObject* CreateBox(const Vector3& pos, const Vector3& rot, const float& w, const float& h, const float& d, const Vector3& su, const Vector3& sv, const int& matId=0, const bool& addToPhysicWorld=true, const bool& isLevelElement=false);

	// creates a mesh
	RenderObject* CreateMesh(const char* fileName, const Vector3& pos, const Vector3& rot, const int& matId=0, const bool& addToPhysicWorld=true, const bool& isLevelElement=false);

	// returns a new/free key
	const unsigned int GetFreeKey(void);

	// flags if materials should be set or not
	void SkipMaterials(bool skip) { skipMaterials = skip; RenderNode::SkipMaterials(skip); }

private:
	// updates the tree
	void UpdateTreeNode(RenderNode* node, float deltaTime, const Matrix4& matrix);

	// renders the tree
	void RenderTreeNode(RenderNode* node);

	// destroy the tree
	void DestroyRenderTree(RenderNode* node);

	map<int, RenderObject*> renderObjects;
	map<int, RenderObject*>::iterator iter;

	RenderNode* root;

	// the skybox textures
	unsigned int skyBoxTextures[6];

	// counters
	unsigned int keyCounter;
	unsigned int uniqueCounter;

	// flags materials are used
	bool skipMaterials;
};

#endif