#ifndef FINAL_PROCESSOR_H
#define FINAL_PROCESSOR_H

#include <IPOST_PROCESSOR.h>

class DX11_RENDER_TARGET;
class DX11_SHADER;

// FINAL_PROCESSOR
//   Copies content of the accumulation buffer (of the GBuffer) into the back buffer.   
class FINAL_PROCESSOR: public IPOST_PROCESSOR
{
public: 
	FINAL_PROCESSOR()
	{
		strcpy(name,"finalProcessor");
		sceneRT = NULL;
		backBufferRT = NULL;
		finalPassShader = NULL;
	}

	virtual bool Create();

	virtual DX11_RENDER_TARGET* GetOutputRT() const;

	virtual void AddSurfaces();

private:
	DX11_RENDER_TARGET *sceneRT;
	DX11_RENDER_TARGET *backBufferRT;
	DX11_SHADER *finalPassShader;

};

#endif