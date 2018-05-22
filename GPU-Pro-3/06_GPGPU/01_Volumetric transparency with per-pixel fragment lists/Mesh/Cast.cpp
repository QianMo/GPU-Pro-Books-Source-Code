#include "DXUT.h"
#include "Mesh/Cast.h"

Mesh::Cast::Cast()
{
}

Mesh::Cast::~Cast()
{
}

void Mesh::Cast::add(Role role, Shaded::P shaded)
{
	RoleShadedMap::iterator i = roleShadedMap.find(role);
	if(i != roleShadedMap.end())
		roleShadedMap.erase(i);
	roleShadedMap[role] = shaded;
}

void Mesh::Cast::draw(ID3D11DeviceContext* context, Role role)
{
	RoleShadedMap::iterator i = roleShadedMap.find(role);
	if(i != roleShadedMap.end())
	{
		i->second->draw(context);
	}
	else
	{
		// warning. could be intentional
	}
}