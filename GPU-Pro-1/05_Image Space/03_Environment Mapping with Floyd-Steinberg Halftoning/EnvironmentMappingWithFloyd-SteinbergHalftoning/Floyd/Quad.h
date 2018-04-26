#pragma once
#include "Cueable.h"

class Theatre;
class XMLNode;
class ShadedMesh;

class Quad :
	public Cueable
{
	ShadedMesh* shadedMesh;

public:
	Quad(Theatre* theatre, XMLNode& xMainNode);
	~Quad(void);

	void render(const RenderContext& context);
	void animate(double dt, double t);
	void control(const ControlContext& context);
	void processMessage( const MessageContext& context);
	Camera* getCamera();
	Node* getInteractors();
};
