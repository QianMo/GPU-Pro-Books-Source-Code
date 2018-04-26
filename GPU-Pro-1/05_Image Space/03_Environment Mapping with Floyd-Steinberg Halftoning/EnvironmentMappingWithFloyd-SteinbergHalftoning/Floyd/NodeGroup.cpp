#include "DXUT.h"
#include "NodeGroup.h"

NodeGroup::~NodeGroup(void)
{
	std::vector<Node*>::iterator i = subnodes.begin();
	while(i != subnodes.end())
	{
		delete (*i);
		i++;
	}
}

void NodeGroup::add(Node* e)
{
	subnodes.push_back(e);
}

void NodeGroup::render(const RenderContext& context)
{
	if(!visible)
		return;
	std::vector<Node*>::iterator i = subnodes.begin();
	while(i != subnodes.end())
	{
		(*i)->render(context);
		i++;
	}
}

void NodeGroup::animate(double dt)
{
	if(!animated)
		return;

	std::vector<Node*>::iterator i = subnodes.begin();
	while(i != subnodes.end())
	{
		(*i)->animate(dt);
		i++;
	}
}

void NodeGroup::control(const ControlContext& context)
{
	std::vector<Node*>::iterator i = subnodes.begin();
	while(i != subnodes.end())
	{
		(*i)->control(context);
		i++;
	}
}

void NodeGroup::interact(Entity* target)
{
	std::vector<Node*>::iterator i = subnodes.begin();
	while(i != subnodes.end())
	{
		(*i)->interact(target);
		i++;
	}
}
