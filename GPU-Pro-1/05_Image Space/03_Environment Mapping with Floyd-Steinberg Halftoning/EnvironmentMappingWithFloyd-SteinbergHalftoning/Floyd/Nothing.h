#pragma once
#include "cueable.h"
#include "Role.h"

class Theatre;
class XMLNode;
class ShadedMesh;

class Nothing :
	public Cueable
{
	class NothingRendition{
		friend class Nothing;
		ID3D10EffectTechnique* technique;
		ID3D10InputLayout* inputLayout;
		unsigned int vertexCount;
	};
	typedef std::map<const Role, NothingRendition> NothingRenditionDirectory;
	NothingRenditionDirectory nothingRenditionDirectory;

public:
	Nothing(Theatre* theatre, XMLNode& xMainNode);
	~Nothing(void);

	void render(const RenderContext& context);
	void animate(double dt, double t);
	void control(const ControlContext& context);
	void processMessage( const MessageContext& context);
	Camera* getCamera();
	Node* getInteractors();
	void loadNothingRenditions(Theatre* theatre, XMLNode& xMainNode);
};
