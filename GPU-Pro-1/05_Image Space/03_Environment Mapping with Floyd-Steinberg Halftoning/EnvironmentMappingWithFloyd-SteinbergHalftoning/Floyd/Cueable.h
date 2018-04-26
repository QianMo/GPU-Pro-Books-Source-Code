#pragma once

#include "Invoke.h"

class RenderContext;
class ControlContext;
class MessageContext;
class Camera;
class Node;
class Theatre;
class Invoke;
class XMLNode;
class ResourceOwner;

class Cueable
{
	Theatre* theatre;
public:
	virtual Invoke* createInvocation(XMLNode& invocationNode, ResourceOwner* localResourceOwner){return NULL;}

	virtual void render(const RenderContext& context)=0;
	virtual void animate(double dt, double t)=0;
	virtual void control(const ControlContext& context)=0;
	virtual void processMessage( const MessageContext& context)=0;
	virtual Camera* getCamera(){return NULL;}
	virtual Node* getInteractors(){return NULL;}
	virtual void assumeAspect(){}

	Cueable(Theatre* theatre){this->theatre = theatre;}
	virtual ~Cueable(){}
	Theatre* getTheatre(){return theatre;}
};