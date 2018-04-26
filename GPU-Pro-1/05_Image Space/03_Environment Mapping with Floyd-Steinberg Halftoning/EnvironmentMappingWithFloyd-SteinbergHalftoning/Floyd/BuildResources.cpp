#include "DXUT.h"
#include "BuildResources.h"
#include "ResourceBuilder.h"
#include "Theatre.h"
#include "xmlParser.h"
#include "TaskContext.h"

BuildResources::BuildResources(XMLNode& resourcesNode, bool swapChainBound)
{
	resourceBuilder = new ResourceBuilder(resourcesNode, swapChainBound);
}

BuildResources::~BuildResources()
{
	delete resourceBuilder;
}

void BuildResources::execute(const TaskContext& context)
{
	resourceBuilder->defineVariables(context.localResourceOwner->getResourceSet(), context.theatre->getDevice());

	resourceBuilder->instantiate(context.localResourceOwner->getResourceSet(), context.theatre->getDevice());
}