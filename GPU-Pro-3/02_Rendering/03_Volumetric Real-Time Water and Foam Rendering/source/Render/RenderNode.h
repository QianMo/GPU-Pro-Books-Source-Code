#ifndef __RENDERNODE__H__
#define __RENDERNODE__H__

#include <map>
#include <list>

#include "../Util/Matrix4.h"

using namespace std;

class RenderObject;

class RenderNode
{
public:
	RenderNode(void);
	~RenderNode(void);

	// sets a matrix id (needed to get the matrix from the physics system)
	void SetMatrixId(const int& id) { matrixId = id; }

	// sets a matrix for the node
	void SetMatrix(const Matrix4& mat) { nodeMatrix = mat; }

	// returns the matrix of the node
	const Matrix4 GetMatrix(void);

	// sets the parent node for this node
	void SetParent(RenderNode* parent);

	// adds a node as child for this node
	void AddChild(RenderNode* child);

	// adds a renderable element to the node
	void AddElement(RenderObject* element);

	// updates the elements of the node
	void Update(float deltaTime, const Matrix4& matrix);

	// renders the elements of the node
	void RenderElements(void);

	// returns the list with the childs of the node
	list<RenderNode*> GetChilds(void) const { return childNodes; }

	// returns an iterator to the begin of the child list
	list<RenderNode*>::iterator GetChildBegin(void) { return childNodes.begin(); }

	// returns an iterator to the end of the child list
	list<RenderNode*>::iterator GetChildEnd(void) { return childNodes.end(); }

	// destroy the node
	void Destroy(void);

	// destroy the child list
	void DestroyChildList(void);

	// sets the nodes name
	void SetNodeName(const char* name) { nodeName = name; }

	// returns the nodes name
	const char* GetNodeName(void) const { return nodeName.c_str(); }

	// flags if materials should be set or not
	static void SkipMaterials(bool skip) { skipMaterials = skip; }

private:
	RenderNode* parentNode;
	list<RenderNode*> childNodes;
	list<RenderNode*>::iterator childIter;

	std::string nodeName;

	int matrixId;
	Matrix4 nodeMatrix;

	multimap<int, RenderObject*> renderObjects;
	multimap<int, RenderObject*>::iterator iter;

	static bool skipMaterials;
};

#endif