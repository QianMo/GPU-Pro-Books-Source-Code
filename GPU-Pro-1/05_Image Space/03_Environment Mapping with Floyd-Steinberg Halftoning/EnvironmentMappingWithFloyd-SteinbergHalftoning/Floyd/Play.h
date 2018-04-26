#pragma once
#include "Directory.h"
#include "ResourceOwner.h"
#include "ResourceBuilder.h"
class XMLNode;
class Theatre;
class PropsMaster;
class ControlStatus;
class MessageContext;
class Role;
class FileFinder;

class Play : public ResourceOwner
{
	Theatre* theatre;

	PropsMaster* propsMaster;
	FileFinder* fileFinder;

	RoleDirectory roleDirectory;

	CueableDirectory cueableDirectory;
	ActDirectory actDirectory;

	ActDirectory::iterator currentAct;

	void loadRoles(XMLNode& playNode);

	void loadCueables(XMLNode& playNode);

	void loadScenes(XMLNode& playNode);
	void loadSceneManagers(XMLNode& sceneNode, SceneManagerList& sceneManagerList);
	void loadQuads(XMLNode& playNode);
	void loadNothings(XMLNode& playNode);
	void loadAutos(XMLNode& playNode);
	void loadTexts(XMLNode& playNode);

	void loadActs(XMLNode& playNode);
	void loadProps(XMLNode& playNode);
	

	ResourceSet* resourceSet;

	ResourceBuilder* resourceBuilder;
	ResourceBuilder* swapChainResourceBuilder;
public:
	Play(Theatre* theatre);
	void loadPlay(XMLNode& playNode);
	void assumeAspect();
	void dropAspect();
	~Play(void);

	Theatre* getTheatre(){return theatre;}

	inline PropsMaster* getPropsMaster(){return propsMaster;}
	inline FileFinder* getFileFinder(){return fileFinder;}

	ResourceSet* getResourceSet();
	ResourceOwner* getParentResourceOwner(Theatre* theatre);
	void loadResourceBuilders(XMLNode& playNode);

	Cueable* getCueable(const std::wstring& name);

	void processMessage( const MessageContext& context);
	void animate(double dt, double t);
	void render();
	void control(const ControlStatus& status, double dt);

	void switchToNextAct(const MessageContext& context);

	const Role getRole(const std::wstring& name);
	const std::wstring getRoleName(const Role& role);

	const std::wstring getCurrentActName();
};
