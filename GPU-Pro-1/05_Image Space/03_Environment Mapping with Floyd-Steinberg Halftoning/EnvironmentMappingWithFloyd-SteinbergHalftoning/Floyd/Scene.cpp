#include "DXUT.h"
#include "Scene.h"
#include "NodeGroup.h"
#include "Entity.h"
#include "RenderContext.h"
#include "MessageContext.h"
#include "FreeCamera.h"
#include "EntityCamera.h"
#include "xmlParser.h"
#include "Theatre.h"
#include "PropsMaster.h"
#include "stlConvert.h"
#include "ShadedEntity.h"
#include "SceneManager.h"

Scene::Scene(Theatre* theatre, SceneManagerList& sceneManagerList, XMLNode& xMainNode)
:Cueable(theatre)
{
	XMLNode& freeCameraNode = xMainNode.getChildNode(L"FreeCamera");
	if(!freeCameraNode.isEmpty())
	{
		cameraDirectory[L"\4D71DXUTFirstPersonCamera"] = new FreeCamera( 
			freeCameraNode.readVector(L"position", D3DXVECTOR3(4, 4, 4)),
			freeCameraNode.readVector(L"lookAt", D3DXVECTOR3(0, 0, 0)));
	}
	else
//	cameraDirectory[L"\4D71DXUTFirstPersonCamera"] = new FreeCamera( D3DXVECTOR3(1, 10, 0), D3DXVECTOR3(0, 20, 0));
//	cameraDirectory[L"\4D71DXUTFirstPersonCamera"] = new FreeCamera( D3DXVECTOR3(-4.0, 11, 7.0), D3DXVECTOR3(0, 0, 0));
//	cameraDirectory[L"\4D71DXUTFirstPersonCamera"] = new FreeCamera( D3DXVECTOR3(1.0, -0.5, -1.0), D3DXVECTOR3(0, -0.7, 0));
//	cameraDirectory[L"\4D71DXUTFirstPersonCamera"] = new FreeCamera( D3DXVECTOR3(-1.0, 0.35, -1.0), D3DXVECTOR3(-1, 0, 0));
//	cameraDirectory[L"\4D71DXUTFirstPersonCamera"] = new FreeCamera( D3DXVECTOR3(1, 0, 0), D3DXVECTOR3(0, 0, 0));
//	cameraDirectory[L"\4D71DXUTFirstPersonCamera"] = new FreeCamera( D3DXVECTOR3(-7, 7, -11), D3DXVECTOR3(0, 0, 0));
//rabbit	cameraDirectory[L"\4D71DXUTFirstPersonCamera"] = new FreeCamera( D3DXVECTOR3(4, 4, -11), D3DXVECTOR3(0, 1.8, 0));
	cameraDirectory[L"\4D71DXUTFirstPersonCamera"] = new FreeCamera( D3DXVECTOR3(4, 4, -11), D3DXVECTOR3(0, 1.8, 0));
//	cameraDirectory[L"\4D71DXUTFirstPersonCamera"] = new FreeCamera( D3DXVECTOR3(-0.85, -0.35, -0.35), D3DXVECTOR3(0, -0.35, 0));
	currentCamera = cameraDirectory.begin();

	SceneManagerList::iterator i = sceneManagerList.begin();
	while(i != sceneManagerList.end())
	{
		(*i)->setScene(this);
		i++;
	}

	loadScene(sceneManagerList, xMainNode);

	i = sceneManagerList.begin();
	while(i != sceneManagerList.end())
	{
		(*i)->finish();
		i++;
	}
}

Scene::~Scene()
{
	cameraDirectory.deleteAll();
	delete worldRoot;
}


void Scene::loadCameras(XMLNode& xMainNode)
{
	int iCamera = 0;
	XMLNode cameraNode;
	while( !(cameraNode = xMainNode.getChildNode(L"Camera", iCamera)).isEmpty() )
	{
		D3DXVECTOR3 position = cameraNode.readVector(L"eye");
		D3DXVECTOR3 lookAt = cameraNode.readVector(L"lookAt", position + D3DXVECTOR3(1, 0, 0));
		D3DXVECTOR3 up = cameraNode.readVector(L"up", D3DXVECTOR3(0, 1, 0));

		double fov = cameraNode.readDouble(L"fov", 1.55);
		double fp = cameraNode.readDouble(L"front", 0.01);
		double bp = cameraNode.readDouble(L"back", 1000.0);

		const wchar_t* ownerEntityName = cameraNode|L"owner";
		if(ownerEntityName != NULL)
		{
			EntityDirectory::iterator iEntity = entityDirectory.find(ownerEntityName);
			if(iEntity != entityDirectory.end())
			{
				double aspect = 1;
				EntityCamera* camera = new EntityCamera(iEntity->second, position, lookAt, up, fov, aspect, fp, bp);
				const wchar_t* cameraName = cameraNode|L"name";
				if(cameraName == NULL)
				{
					std::wstring autoName(L"engineNamedAutomaticCamera");
					autoName += std::toWString<size_t>(cameraDirectory.size());
					cameraDirectory[autoName] = camera;
				}
				else
					if(cameraDirectory[cameraName] == NULL)
						cameraDirectory[cameraName] = camera;
					else
						EggXMLERR(cameraNode, L"Duplicate Camera name: " << cameraName);
				currentCamera = cameraDirectory.begin();
			}
		}
		iCamera++;
	}
}

void Scene::loadGroup(SceneManagerList& sceneManagerList, XMLNode& groupNode, NodeGroup*& group)
{
	if(groupNode.isEmpty())
		return;

	group = new NodeGroup();

	loadEntities(sceneManagerList, groupNode, group);
}

void Scene::loadEntities(SceneManagerList& sceneManagerList, XMLNode& groupNode, NodeGroup* group)
{
	int iEntity = 0;
	XMLNode entityNode;
	while( !(entityNode = groupNode.getChildNode(L"Entity", iEntity)).isEmpty() )
	{
		const wchar_t* entityName = entityNode|L"name";
		Entity* entity = NULL;
		int iDecorator = 0;
		XMLNode decoratorNode;
		while( !(decoratorNode = entityNode.getChildNode(iDecorator)).isEmpty() )
		{
			bool processed = false;
			if(wcscmp(decoratorNode.getName(), L"ShadedEntity") == 0)
			{
				processed = true;
				if(entity)
				{
					if(entityName)
					{
						EggXMLERR(entityNode, L"ShadedEntity is not a decorator. It is used as such in entity " << entityName);
					}
					else
						EggXMLERR(entityNode, L"ShadedEntity is not a decorator.");
				}
				ShadedMesh* shadedMesh = getTheatre()->getPropsMaster()->getShadedMesh(decoratorNode|L"shadedMesh");
				if(shadedMesh)
				{
					entity = new ShadedEntity(shadedMesh);
				}
				else
				{
					if(entityName)
					{
						EggXMLERR(entityNode, L"No ShadedMesh specified for ShadedEnity " << entityName);
					}
					else
						EggXMLERR(entityNode, L"No ShadedMesh specified for ShadedEnity.");
				}
			}
			SceneManagerList::iterator i = sceneManagerList.begin();
			while(i != sceneManagerList.end())
			{
				entity = (*i)->decorateEntity(entity, decoratorNode, processed);
				i++;
			}
			if(!processed)
				EggXMLERR(decoratorNode, L"Entity decorator tag not recognized by any SceneManager.");
			iDecorator++;
		}
		if(entityName)
			if(entityDirectory[entityName] == NULL)
				entityDirectory[entityName] = entity;
			else
				EggXMLERR(entityNode, L"Duplicate Entity name: " << entityName);
		group->add(entity);
		iEntity++;
	}
}

void Scene::loadScene(SceneManagerList& sceneManagerList, XMLNode& xMainNode)
{

	XMLNode groupNode = xMainNode.getChildNode(L"Group");
	worldRoot = NULL;
	loadGroup(sceneManagerList, groupNode, worldRoot);
	loadCameras(xMainNode);
}


void Scene::render(const RenderContext& context)
{
	if(worldRoot == NULL)
		return;

	Camera* camera = context.camera?context.camera:currentCamera->second;
	context.theatre->getEffect()->GetVariableByName("viewProjMatrix")->AsMatrix()->SetMatrix(
		(float*)&(camera->getViewMatrix() * camera->getProjMatrix()));

	D3DXMATRIX rootNodeTransform;
	D3DXMatrixIdentity(&rootNodeTransform);
	worldRoot->render(RenderContext(context.theatre, context.localResourceOwner, camera,  &rootNodeTransform, context.role, context.instanceCount));
}

void Scene::animate(double dt, double t)
{
	currentCamera->second->animate(dt);
	worldRoot->animate(dt);
}

void Scene::control(const ControlContext& context)
{
	worldRoot->control(context);
}

void Scene::processMessage( const MessageContext& context)
{
	if(context.uMsg == WM_KEYDOWN)
	{
		if(context.wParam == VK_ADD)
		{
			CameraDirectory::iterator nextCamera = currentCamera;
			nextCamera++;
			if(nextCamera != cameraDirectory.end())
				currentCamera = nextCamera;
		}
		if(context.wParam == VK_SUBTRACT)
			if(currentCamera != cameraDirectory.begin())
				currentCamera--;
	}
	currentCamera->second->handleInput(context.hWnd, context.uMsg, context.wParam, context.lParam);
}

Camera* Scene::getCamera()
{
	return currentCamera->second;
}

Node* Scene::getInteractors()
{
	return this->worldRoot;
}

void Scene::assumeAspect()
{
	D3D10_VIEWPORT vp;
	UINT cRT = 1;
	getTheatre()->getDevice()->RSGetViewports( &cRT, &vp );

	double aspect = (double)vp.Width / vp.Height;

	CameraDirectory::iterator i = cameraDirectory.begin();
	while(i != cameraDirectory.end())
	{

		i->second->setAspect(aspect);
		i++;
	}
}



