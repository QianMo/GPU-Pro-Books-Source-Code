#include "DXUT.h"
#include "Mesh/Bound.h"


Mesh::Bound::Bound( Geometry::P geometry, ID3D11InputLayout* inputLayout)
	:geometry(geometry), inputLayout(inputLayout)
{
	this->inputLayout->AddRef();
}


Mesh::Bound::~Bound(void)
{
	inputLayout->Release();
}

void Mesh::Bound::draw(ID3D11DeviceContext* context)
{
	context->IASetInputLayout(inputLayout);
	geometry->draw(context);
}