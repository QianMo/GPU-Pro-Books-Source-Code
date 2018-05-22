#include "DXUT.h"
#include "Mesh/Shaded.h"


void Mesh::Shaded::draw(ID3D11DeviceContext* context)
{
	material->apply(context);
	bound->draw(context);
}