#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <GL/gl.h>

#include <assert.h>

#include "../Render/RenderNode.h"
#include "../Render/RenderObject.h"
#include "../Render/MaterialManager.h"
#include "../Render/ShaderManager.h"

#include "../Physic/Physic.h"

#include <GL/glut.h>

bool RenderNode::skipMaterials = false;

// -----------------------------------------------------------------------------
// ------------------------ RenderNode::RenderNode -----------------------------
// -----------------------------------------------------------------------------
RenderNode::RenderNode(void):
	matrixId(-1),
	nodeName("")
{
	nodeMatrix = Matrix4::IDENTITY;
}

// -----------------------------------------------------------------------------
// ------------------------ RenderNode::~RenderNode ----------------------------
// -----------------------------------------------------------------------------
RenderNode::~RenderNode(void)
{
}

// -----------------------------------------------------------------------------
// ------------------------ RenderNode::SetParent ------------------------------
// -----------------------------------------------------------------------------
const Matrix4 RenderNode::GetMatrix(void)
{
	if(matrixId >= 0)
		return Physic::Instance()->GetActorsMatrix(matrixId);
	else
		return nodeMatrix;
}

// -----------------------------------------------------------------------------
// ------------------------ RenderNode::SetParent ------------------------------
// -----------------------------------------------------------------------------
void RenderNode::SetParent(RenderNode* parent)
{
	parentNode = parent;
}

// -----------------------------------------------------------------------------
// ------------------------ RenderNode::AddChild -------------------------------
// -----------------------------------------------------------------------------
void RenderNode::AddChild(RenderNode* child)
{
	child->SetParent(this);
	childNodes.push_back(child);
}

// -----------------------------------------------------------------------------
// ------------------------ RenderNode::AddElement -----------------------------
// -----------------------------------------------------------------------------
void RenderNode::AddElement(RenderObject* element)
{
	renderObjects.insert(std::make_pair(element->GetKey().GetIntKey(), element));
}

// -----------------------------------------------------------------------------
// ------------------------ RenderNode::Update ---------------------------------
// -----------------------------------------------------------------------------
void RenderNode::Update(float deltaTime, const Matrix4& matrix)
{
	for (iter = renderObjects.begin(); iter != renderObjects.end(); ++iter)
	{
		RenderObject* object = iter->second;
		object->Update(deltaTime, matrix);
	}
}

// -----------------------------------------------------------------------------
// ------------------------ RenderNode::RenderElements -------------------------
// -----------------------------------------------------------------------------
void RenderNode::RenderElements()
{
	int currentMaterial = 0;

	if(matrixId >= 0)
		glMultMatrixf(Physic::Instance()->GetActorsMatrix(matrixId).entry);
	else
		glMultMatrixf(nodeMatrix.entry);

	for (iter = renderObjects.begin(); iter != renderObjects.end(); ++iter)
	{
		RenderObject* object = iter->second;

		if ((currentMaterial != object->GetMaterialId()) && !skipMaterials)
		{
			currentMaterial = object->GetMaterialId();
			MaterialManager::Instance()->SetMaterial(object->GetMaterialId());

			Material::MaterialDefinition mat = MaterialManager::Instance()->GetMaterial(object->GetMaterialId());
			if (mat.useParallaxMapping)
				ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SURFACE, "ParallaxMapping");
			else
				ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SURFACE, "TextureMapping");
		}

		object->Render();
	}
}

// -----------------------------------------------------------------------------
// ------------------------ RenderNode::Destroy --------------------------------
// -----------------------------------------------------------------------------
void RenderNode::Destroy(void)
{
	assert(childNodes.size() == 0);
	for (iter = renderObjects.begin(); iter != renderObjects.end(); ++iter)
	{
		RenderObject* object = iter->second;
		delete object;
		object = NULL;
	}
	renderObjects.erase(renderObjects.begin(), renderObjects.end());
}

// -----------------------------------------------------------------------------
// ------------------------ RenderNode::Destroy --------------------------------
// -----------------------------------------------------------------------------
void RenderNode::DestroyChildList(void)
{
	childNodes.erase(childNodes.begin(), childNodes.end());
}