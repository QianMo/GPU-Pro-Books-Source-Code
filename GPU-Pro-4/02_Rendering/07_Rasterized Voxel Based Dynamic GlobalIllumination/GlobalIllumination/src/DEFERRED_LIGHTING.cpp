#include <stdafx.h>
#include <DEMO.h>
#include <DEFERRED_LIGHTING.h>

bool DEFERRED_LIGHTING::Create()
{
	return true;
}

DX11_RENDER_TARGET* DEFERRED_LIGHTING::GetOutputRT() const
{
	return NULL;
}

void DEFERRED_LIGHTING::AddSurfaces()
{
	for(int i=0;i<DEMO::renderer->GetNumLights();i++) 
	{
		ILIGHT *light = DEMO::renderer->GetLight(i);
		if(light->IsActive())
			light->AddLitSurface();
	}
}
