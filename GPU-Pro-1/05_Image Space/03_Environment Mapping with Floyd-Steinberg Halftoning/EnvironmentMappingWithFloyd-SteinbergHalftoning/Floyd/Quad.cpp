#include "DXUT.h"
#include "Quad.h"
#include "Theatre.h"
#include "PropsMaster.h"
#include "ShadedMesh.h"
#include "Camera.h"
#include "RenderContext.h"
#include "XMLparser.h"

Quad::Quad(Theatre* theatre, XMLNode& xMainNode)
:Cueable(theatre)
{
	PropsMaster* propsMaster = theatre->getPropsMaster();
	shadedMesh = new ShadedMesh(propsMaster->getMesh(L"\4D71quad"));
	propsMaster->createShadingMaterials(xMainNode, shadedMesh);
}

Quad::~Quad(void)
{
	delete shadedMesh;
}

void Quad::render(const RenderContext& context)
{
	if(context.camera)
	{
		context.theatre->getEffect()->GetVariableByName("orientProjMatrixInverse")->AsMatrix()->SetMatrix(
			(float*)&context.camera->getOrientProjMatrixInverse());
		context.theatre->getEffect()->GetVariableByName("viewProjMatrix")->AsMatrix()->SetMatrix(
			(float*)&(context.camera->getViewMatrix() * context.camera->getProjMatrix()));
		context.theatre->getEffect()->GetVariableByName("eyePosition")->AsVector()->SetFloatVector(
			(float*)&context.camera->getEyePosition());
	}

	shadedMesh->render(context);
}

void Quad::animate(double dt, double t)
{}

void Quad::control(const ControlContext& context)
{}

void Quad::processMessage( const MessageContext& context)
{}

Camera* Quad::getCamera()
{
	return NULL;
}

Node* Quad::getInteractors()
{
	return NULL;
}