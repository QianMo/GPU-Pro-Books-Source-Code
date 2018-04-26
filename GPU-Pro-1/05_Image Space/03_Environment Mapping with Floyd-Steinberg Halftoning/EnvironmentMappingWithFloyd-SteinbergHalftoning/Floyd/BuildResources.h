#pragma once
#include "task.h"

class ResourceBuilder;
class XMLNode;

class BuildResources :
	public Task
{
	ResourceBuilder* resourceBuilder;
public:
	BuildResources(XMLNode& resourcesNode, bool swapChainBound);
	~BuildResources();

	void execute(const TaskContext& context);
};
