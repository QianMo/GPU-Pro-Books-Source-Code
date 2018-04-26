#include "DXUT.h"
#include "Role.h"

unsigned int Role::nextId(0);

const Role Role::invalid;

Role::Role()
{
	id = nextId;
	nextId++;
}

Role::Role(const Role& o)
{
	id = o.id;
}

bool Role::operator<(const Role& o) const
{
	return id < o.id;
}

bool Role::operator==(const Role& o) const
{
	return id == o.id;
}