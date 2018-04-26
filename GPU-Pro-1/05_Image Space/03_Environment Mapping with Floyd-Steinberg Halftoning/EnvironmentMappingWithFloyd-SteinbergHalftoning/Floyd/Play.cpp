#include "DXUT.h"
#include "Play.h"
#include "Theatre.h"
#include "Act.h"
#include "Cueable.h"
#include "Scene.h"
#include "Quad.h"
#include "Auto.h"
#include "Nothing.h"
#include "Text.h"
#include "XMLparser.h"
#include "PropsMaster.h"
#include "FileFinder.h"
#include "ControlContext.h"
#include "ResourceSet.h"
#include "MessageContext.h"
#include "Role.h"
#include "ScriptVariableClass.h"
#include "ScriptDepthStencilViewVariable.h"
#include "ScriptRenderTargetViewVariable.h"
#include "StaticScene.h"
#include "RaytracingScene.h"


Play::Play(Theatre* theatre)
{
	this->theatre = theatre;
	resourceSet = new ResourceSet();
	resourceBuilder = NULL;
	swapChainResourceBuilder = NULL;
}

void Play::loadPlay(XMLNode& playNode)
{
	fileFinder = new FileFinder(playNode);
	loadRoles(playNode);
	loadResourceBuilders(playNode);
	loadTaskBlocks(this, playNode);
	if(resourceBuilder)
	{
		resourceBuilder->defineVariables(resourceSet, theatre->getDevice());
		resourceBuilder->instantiate(resourceSet, theatre->getDevice());
	}
	if(swapChainResourceBuilder)
	{
		swapChainResourceBuilder->defineVariables(resourceSet, theatre->getDevice());
	}
	resourceSet->createVariable(ScriptVariableClass::DepthStencilView, L"default");
	resourceSet->createVariable(ScriptVariableClass::RenderTargetView, L"default");
	propsMaster= new PropsMaster(theatre);
	loadProps(playNode);
	loadCueables(playNode);
	loadActs(playNode);
}

Play::~Play(void)
{
	resourceSet->releaseResources();
	{
		if(resourceBuilder)
			delete resourceBuilder;
	}
	{
		if(swapChainResourceBuilder)
			delete swapChainResourceBuilder;
	}
	delete resourceSet;
	cueableDirectory.deleteAll();
	actDirectory.deleteAll();
	delete propsMaster;
	delete fileFinder;
}

void Play::loadProps(XMLNode& playNode)
{
	int iProps = 0;
	XMLNode propsNode;
	while( !(propsNode = playNode.getChildNode(L"addProps", iProps)).isEmpty() )
	{
		const wchar_t* file = propsNode|L"file";
		if(file)
		{
			XMLNode propsNode = fileFinder->openXMLFile(file, L"props");
			if(!propsNode.isEmpty())
				propsMaster->addProps(propsNode);
		}
		iProps++;
	}
}

void Play::loadCueables(XMLNode& playNode)
{
	loadScenes(playNode);
	loadQuads(playNode);
	loadNothings(playNode);
	loadAutos(playNode);
	loadTexts(playNode);
}

void Play::loadScenes(XMLNode& playNode)
{
	int iScene = 0;
	XMLNode sceneNode;
	while( !(sceneNode = playNode.getChildNode(L"Scene", iScene)).isEmpty() )
	{
		const wchar_t* cue = sceneNode|L"cue";
		const wchar_t* file = sceneNode|L"file";
		if(cue && file)
		{
			XMLNode sceneFileNode = fileFinder->openXMLFile(file, L"Scene");
			if(!sceneFileNode.isEmpty())
			{
				int nManagers = sceneNode.nChildNode();
				SceneManagerList sceneManagerList;
				sceneManagerList.reserve(nManagers);
				loadSceneManagers(sceneNode, sceneManagerList);
				cueableDirectory[cue] = new Scene(theatre, sceneManagerList, sceneFileNode);
			}
			else
				EggERR(L"Scene file not found. [" << file << L"]");
		}
		else
			EggXMLERR(playNode, L"Scene without cue or file.");
		iScene++;
	}
}

void Play::loadSceneManagers(XMLNode& sceneNode, SceneManagerList& sceneManagerList)
{
	int iSceneManager = 0;
	XMLNode sceneManagerNode;
	while( !(sceneManagerNode = sceneNode.getChildNode(iSceneManager)).isEmpty() )
	{
		const wchar_t* cue = sceneManagerNode|L"cue";
		if(cue)
		{
			if(wcscmp(sceneManagerNode.getName(), L"StaticScene") == 0)
			{
				StaticScene* staticScene = new StaticScene(theatre, sceneManagerNode);
				sceneManagerList.push_back(staticScene);
				cueableDirectory[cue] = staticScene;
			}
			else if(wcscmp(sceneManagerNode.getName(), L"RaytracingScene") == 0)
			{
				RaytracingScene* raytracingScene = new RaytracingScene(theatre, sceneManagerNode);
				sceneManagerList.push_back(raytracingScene);
				cueableDirectory[cue] = raytracingScene;
			}
		}
		else
			EggXMLERR(sceneManagerNode, L"No cue for a decorator.");
		iSceneManager++;
	}
}

void Play::loadQuads(XMLNode& playNode)
{
	int iQuad = 0;
	XMLNode quadNode;
	while( !(quadNode = playNode.getChildNode(L"Quad", iQuad)).isEmpty() )
	{
		const wchar_t* cue = quadNode|L"cue";
		const wchar_t* file = quadNode|L"file";
		if(cue && file)
		{
			XMLNode quadNode = fileFinder->openXMLFile(file, L"Quad");
			if(!quadNode.isEmpty())
				cueableDirectory[cue] = new Quad(theatre, quadNode);
		}
		iQuad++;
	}
}

void Play::loadNothings(XMLNode& playNode)
{
	int iNothing = 0;
	XMLNode nothingNode;
	while( !(nothingNode = playNode.getChildNode(L"Nothing", iNothing)).isEmpty() )
	{
		const wchar_t* cue = nothingNode|L"cue";
		const wchar_t* file = nothingNode|L"file";
		if(cue && file)
		{
			XMLNode nothingNode = fileFinder->openXMLFile(file, L"Nothing");
			if(!nothingNode.isEmpty())
				cueableDirectory[cue] = new Nothing(theatre, nothingNode);
		}
		iNothing++;
	}
}

void Play::loadAutos(XMLNode& playNode)
{
	int iAuto = 0;
	XMLNode autoNode;
	while( !(autoNode = playNode.getChildNode(L"Auto", iAuto)).isEmpty() )
	{
		const wchar_t* cue = autoNode|L"cue";
		const wchar_t* file = autoNode|L"file";
		if(cue && file)
		{
			XMLNode autoNode = fileFinder->openXMLFile(file, L"Auto");
			if(!autoNode.isEmpty())
				cueableDirectory[cue] = new Auto(theatre, autoNode);
		}
		iAuto++;
	}
}


void Play::loadTexts(XMLNode& playNode)
{
	int iText = 0;
	XMLNode textNode;
	while( !(textNode = playNode.getChildNode(L"Text", iText)).isEmpty() )
	{
		const wchar_t* cue = textNode|L"cue";
		const wchar_t* file = textNode|L"file";
		if(cue && file)
		{
			XMLNode textNode = fileFinder->openXMLFile(file, L"Text");
			if(!textNode.isEmpty())
				cueableDirectory[cue] = new Text(theatre, textNode);
		}
		iText++;
	}
}

void Play::loadActs(XMLNode& playNode)
{
	int iAct = 0;
	XMLNode actNode;
	while( !(actNode = playNode.getChildNode(L"Act", iAct)).isEmpty() )
	{
		const wchar_t* title = actNode|L"title";
		const wchar_t* disabled = actNode|L"disabled";
		if(title && disabled == NULL)
			actDirectory[title] = new Act(this, actNode);
		iAct++;
	}
	currentAct = actDirectory.begin();
}

void Play::processMessage(const MessageContext& context)
{
	if(context.uMsg == WM_SIZE)
	{
		MessageContext initContext = context;
		initContext.uMsg = WM_NULL;
		initContext.lParam = Theatre::actActivate;
		currentAct->second->processMessage(initContext);
	}

	if(context.uMsg == WM_KEYDOWN)
	{
		if(context.wParam == VK_SPACE)
			switchToNextAct(context);
	}
	/*
	ActDirectory::iterator iAct = actDirectory.begin();
	while(iAct != actDirectory.end())
	{
		iAct->second->processMessage(hWnd, uMsg, wParam, lParam);
		iAct++;
	}
	*/
	currentAct->second->processMessage(context);
}

void Play::animate(double dt, double t)
{
/*	ActDirectory::iterator iAct = actDirectory.begin();
	while(iAct != actDirectory.end())
	{
		iAct->second->animate(dt, t);
		iAct++;
	}*/
	currentAct->second->animate(dt, t);
}

void Play::control(const ControlStatus& status, double dt)
{
/*	ActDirectory::iterator iAct = actDirectory.begin();
	while(iAct != actDirectory.end())
	{
		iAct->second->control(status, dt);
		iAct++;
	}*/
	currentAct->second->control(status, dt);
}

void Play::render()
{
/*	ActDirectory::iterator iAct = actDirectory.begin();
	while(iAct != actDirectory.end())
	{
		iAct->second->render();
		iAct++;
	}*/
	currentAct->second->render();
}

Cueable* Play::getCueable(const std::wstring& name)
{
	CueableDirectory::iterator iCueable = cueableDirectory.find(name);
	if(iCueable != cueableDirectory.end())
		return iCueable->second;
	else
	{
		EggERR(L"Unknown cue. [" << name << "}");
		return NULL;
	}
}


void Play::assumeAspect()
{
    ID3D10RenderTargetView* defaultRenderTargetView;
    ID3D10DepthStencilView* defaultDepthStencilView;
	theatre->getDevice()->OMGetRenderTargets( 1, &defaultRenderTargetView, &defaultDepthStencilView );

	resourceSet->getRenderTargetViewVariable(L"default")->setRenderTargetView(defaultRenderTargetView);
	resourceSet->getDepthStencilViewVariable(L"default")->setDepthStencilView(defaultDepthStencilView);

	CueableDirectory::iterator iCueable = cueableDirectory.begin();
	while(iCueable != cueableDirectory.end())
	{
		iCueable->second->assumeAspect();
		iCueable++;
	}
	ResourceBuilder* resourceBuilder = swapChainResourceBuilder;
	if(resourceBuilder)
		resourceBuilder->instantiate(resourceSet, theatre->getDevice());
}

void Play::dropAspect()
{
	resourceSet->getRenderTargetViewVariable(L"default")->releaseResource();
	resourceSet->getDepthStencilViewVariable(L"default")->releaseResource();
	resourceSet->releaseSwapChainResources();
}

ResourceSet* Play::getResourceSet()
{
	return resourceSet;
}

void Play::loadResourceBuilders(XMLNode& playNode)
{
	int iResourceBuilder = 0;
	XMLNode resourceBuilderNode;
	while( !(resourceBuilderNode = playNode.getChildNode(L"ResourceBuilder", iResourceBuilder)).isEmpty() )
	{
		const wchar_t* instantiationEvent = resourceBuilderNode|L"instantiationEvent";
		bool swapChainBound = (wcscmp(instantiationEvent, L"resize") == 0);
		ResourceBuilder*& rbd = 
			swapChainBound?
			swapChainResourceBuilder:
			resourceBuilder;
	
		if(rbd)
		{
			EggERR(L"Multiple resource builders.");
			delete rbd;
		}

		rbd = new ResourceBuilder(resourceBuilderNode, swapChainBound);

		iResourceBuilder++;
	}
}

void Play::switchToNextAct(const MessageContext& context)
{
	ActDirectory::iterator iNextAct = currentAct;
	iNextAct++;
	if(iNextAct == actDirectory.end())
		currentAct = actDirectory.begin();
	else
		currentAct = iNextAct;
	MessageContext initContext = context;
	initContext.uMsg = WM_NULL;
	initContext.lParam = Theatre::actActivate;
	currentAct->second->processMessage(initContext);
}

void Play::loadRoles(XMLNode& playNode)
{
	int iRole = 0;
	XMLNode roleNode;
	while( !(roleNode = playNode.getChildNode(L"Role", iRole)).isEmpty() )
	{
		const wchar_t* name = roleNode|L"name";
		if(name)
		{
			roleDirectory.insert(std::pair<const std::wstring, const Role>(name, Role()));
		}
		else
			EggXMLERR(roleNode, L"No name specified for Role.");
		iRole++;
	}
}

const Role Play::getRole(const std::wstring& name)
{
	RoleDirectory::iterator i = roleDirectory.find(name);
	if(i != roleDirectory.end())
		return i->second;
	else
	{
		return Role::invalid;
	}
}

const std::wstring Play::getRoleName(const Role& role)
{
	RoleDirectory::iterator i = roleDirectory.begin();
	RoleDirectory::iterator e = roleDirectory.end();
	while(i != e)
	{
		if(i->second == role)
			return i->first;
		i++;
	}
	return L"Unknown role.";
}

ResourceOwner* Play::getParentResourceOwner(Theatre* theatre)
{
	return NULL;
}

const std::wstring Play::getCurrentActName()
{
	return currentAct->first;
}