#pragma once
#include "cueable.h"

class Theatre;
class XMLNode;
class Entity;

class SceneManager :
	public Cueable
{
	friend class Scene;
	void setScene(Scene* scene){this->scene = scene;}
	virtual Entity* decorateEntity(Entity* entity, XMLNode& entityNode, bool& processed)=0;
	virtual void finish()=0;

protected:
	Scene* scene;
public:
	SceneManager(Theatre* theatre);
	virtual ~SceneManager();

	virtual void render(const RenderContext& context);
	virtual void animate(double dt, double t);
	virtual void control(const ControlContext& context);
	virtual void processMessage( const MessageContext& context);
	virtual Camera* getCamera(){return NULL;}
	virtual Node* getInteractors(){return NULL;}
	virtual void assumeAspect(){}
};
