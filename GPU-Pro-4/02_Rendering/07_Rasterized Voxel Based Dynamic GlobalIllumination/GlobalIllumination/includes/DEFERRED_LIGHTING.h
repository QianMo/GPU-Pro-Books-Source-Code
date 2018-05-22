#ifndef DEFERRED_LIGHTING_H
#define DEFERRED_LIGHTING_H

#include <IPOST_PROCESSOR.h>

// DEFERRED_LIGHTING
//   Performs deferred lighting.
class DEFERRED_LIGHTING: public IPOST_PROCESSOR
{
public: 
	DEFERRED_LIGHTING()
	{
		strcpy(name,"DeferredLighting");
	}

	virtual bool Create();

	virtual DX11_RENDER_TARGET* GetOutputRT() const;

	virtual void AddSurfaces();

};

#endif