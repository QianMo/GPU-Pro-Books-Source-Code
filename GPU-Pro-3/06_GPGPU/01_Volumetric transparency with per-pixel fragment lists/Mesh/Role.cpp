#include "DXUT.h"
#include "Mesh/Role.h"


unsigned int Mesh::Role::nextId(0);

const Mesh::Role Mesh::Role::invalid;

Mesh::Role::Role()
{
	id = nextId;
	nextId++;
}

bool Mesh::Role::operator<(const Role& co) const
{
	return id < co.id;
}

bool Mesh::Role::operator==(const Role& co) const
{
	return id == co.id;
}