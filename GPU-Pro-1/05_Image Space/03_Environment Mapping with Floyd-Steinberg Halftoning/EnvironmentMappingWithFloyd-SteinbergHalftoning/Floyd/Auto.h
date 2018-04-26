#pragma once
#include "cueable.h"
#include "Role.h"

class Theatre;
class XMLNode;
class ShadedMesh;

class Auto :
	public Cueable
{
	class AutoRendition{
		friend class Auto;
		ID3D10EffectTechnique* technique;
		ID3D10InputLayout* inputLayout;
	};
	typedef std::map<const Role, AutoRendition> AutoRenditionDirectory;
	AutoRenditionDirectory autoRenditionDirectory;

public:
	Auto(Theatre* theatre, XMLNode& xMainNode);
	~Auto(void);

	void render(const RenderContext& context);
	void animate(double dt, double t);
	void control(const ControlContext& context);
	void processMessage( const MessageContext& context);
	Camera* getCamera();
	Node* getInteractors();
	void loadAutoRenditions(Theatre* theatre, XMLNode& xMainNode);
};
