#pragma once
#include "Cueable.h"
#include "Directory.h"

class XMLNode;
class NodeGroup;
class PhysicsEntity;
class Theatre;
class RaytracingMesh;

class Scene : public Cueable
{
	/// Entity references accessible by unicode name.
	EntityDirectory		entityDirectory;

	/// Camera references accessible by unicode name.
	CameraDirectory		cameraDirectory;

	/// Iterator pointing to current camera;
	CameraDirectory::iterator currentCamera;

	/// Processes all 'camera' tags in the XML node, creating EntityCamera instances.
	void loadCameras(XMLNode& xMainNode);

	/// Builds a scene from an XML file.
	void loadScene(SceneManagerList& sceneManagerList, XMLNode& xMainNode);

	/// Loads an entity group from an XML file, building a subtree of the theatre graph.
	void loadGroup(SceneManagerList& sceneManagerList, XMLNode& groupNode, NodeGroup*& group);

	void loadEntities(SceneManagerList& sceneManagerList, XMLNode& groupNode, NodeGroup* group);

	/// Scene graph root reference.
	NodeGroup* worldRoot;

public:
	Scene(Theatre* theatre, SceneManagerList& sceneManagerList, XMLNode& xMainNode);
	~Scene();

	void assumeAspect();

	void render(const RenderContext& context);
	void animate(double dt, double t);
	void control(const ControlContext& context);
	void processMessage( const MessageContext& context);
	Camera* getCamera();
	Node* getInteractors();
};
