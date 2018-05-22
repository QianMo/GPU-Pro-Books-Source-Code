#ifndef SKY_H
#define SKY_H

#include <IPOST_PROCESSOR.h>

class DX11_RENDER_TARGET;
class RENDER_TARGET_CONFIG;
class DX11_SHADER;
class DX11_DEPTH_STENCIL_STATE;

// SKY
//   Extremely simple sky post-processor. Since all previously rendered opaque geometry  
//   had incremented the stencil buffer, for the sky a constant colored full-screen quad 
//   is only rendered where the stencil buffer is still 0.
class SKY: public IPOST_PROCESSOR
{
public: 
	SKY()
	{
		strcpy(name,"SKY");
		sceneRT = NULL;
		rtConfig = NULL;
		skyShader = NULL;
		depthStencilState = NULL;
	}

	virtual bool Create();

	virtual DX11_RENDER_TARGET* GetOutputRT() const;

	virtual void AddSurfaces();

private:
	DX11_RENDER_TARGET *sceneRT;
  RENDER_TARGET_CONFIG *rtConfig;
	DX11_SHADER *skyShader;
	DX11_DEPTH_STENCIL_STATE *depthStencilState;

};

#endif