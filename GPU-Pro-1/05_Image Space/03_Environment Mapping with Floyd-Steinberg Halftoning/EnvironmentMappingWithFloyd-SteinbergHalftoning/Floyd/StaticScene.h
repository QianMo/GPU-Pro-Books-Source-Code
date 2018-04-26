#pragma once
#include "SceneManager.h"

class StaticScene :
	public SceneManager
{
	Entity* decorateEntity(Entity* entity, XMLNode& entityNode, bool& processed);
	void finish();
public:
	StaticScene(Theatre* theatre, XMLNode& staticEntityNode);
};
