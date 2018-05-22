#ifndef RENDER_TARGET_CONFIG_H
#define RENDER_TARGET_CONFIG_H

#define MAX_NUM_SB_BUFFERS 2 // max number of structured buffers that can be set in RENDER_TARGET_CONFIG

class DX11_RENDER_TARGET;
class DX11_STRUCTURED_BUFFER;

// descriptor for setting up RENDER_TARGET_CONFIG
struct RT_CONFIG_DESC
{
	RT_CONFIG_DESC()
	{
	  firstColorBufferIndex = 0;
		numColorBuffers = 1;
		numStructuredBuffers = 0;
		for(int i=0;i<MAX_NUM_SB_BUFFERS;i++)
		  structuredBuffers[i] = NULL;
		computeTarget = false;
	}

	bool operator== (const RT_CONFIG_DESC &desc) const
	{
	  if(firstColorBufferIndex!=desc.firstColorBufferIndex)
			return false;
		if(numColorBuffers!=desc.numColorBuffers)
			return false;
		if(numStructuredBuffers!=desc.numStructuredBuffers)
			return false;
		for(int i=0;i<MAX_NUM_SB_BUFFERS;i++)
		{
			if(structuredBuffers[i]!=desc.structuredBuffers[i])
				return false;
		}
		if(computeTarget!=desc.computeTarget)
			return false;
		return true;
	}

	int firstColorBufferIndex; // index of first render-target to render into 
	int numColorBuffers; // number of render-targets to render into
	int numStructuredBuffers; // number of structured-buffers to write into
	DX11_STRUCTURED_BUFFER *structuredBuffers[8]; // structured-buffers to write into
	bool computeTarget; // true, when using corresponding render-target for compute shader   
};

// RENDER_TARGET_CONFIG
//   Offers the possibility to configure a render-target.
class RENDER_TARGET_CONFIG
{
public:
	bool Create(const RT_CONFIG_DESC &desc)
	{
		this->desc = desc;
		return true;
	}

	RT_CONFIG_DESC GetDesc() const
	{
		return desc;
	}

private:
	RT_CONFIG_DESC desc;

};

#endif