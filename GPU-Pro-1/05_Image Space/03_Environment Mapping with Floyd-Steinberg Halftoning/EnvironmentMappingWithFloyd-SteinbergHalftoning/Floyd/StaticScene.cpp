#include "DXUT.h"
#include "StaticScene.h"
#include "StaticEntity.h"
#include "xmlParser.h"

Entity* StaticScene::decorateEntity(Entity* entity, XMLNode& staticEntityNode, bool& processed)
{
	if(wcscmp(staticEntityNode.getName(), L"StaticEntity")==0)
	{
		NxVec3 pos = staticEntityNode.readNxVec3(L"position");
		NxQuat ori = staticEntityNode.readNxQuat(L"orientation");

		processed = true;
		return new StaticEntity(entity, pos, ori);
	}
	return entity;
}

void StaticScene::finish()
{
}

StaticScene::StaticScene(Theatre* theatre, XMLNode& decoratorNode)
:SceneManager(theatre)
{
}

