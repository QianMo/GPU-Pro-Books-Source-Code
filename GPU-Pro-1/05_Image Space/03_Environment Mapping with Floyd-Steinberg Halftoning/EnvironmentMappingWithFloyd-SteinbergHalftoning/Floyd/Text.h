#pragma once
#include "cueable.h"
#include "SDKmisc.h"

class Theatre;
class RenderContext;
class XMLNode;

class Text :
	public Cueable
{
	// Text objects.
	CDXUTTextHelper* textHelper;
	ID3DX10Font* textFont;
	ID3DX10Sprite* textSprite;

public:
	Text(Theatre* theatre, XMLNode& xMainNode);
	~Text(void);

	void render(const RenderContext& context);
	void animate(double dt, double t);
	void control(const ControlContext& context);
	void processMessage( const MessageContext& context);
	Camera* getCamera();
	Node* getInteractors();
};
