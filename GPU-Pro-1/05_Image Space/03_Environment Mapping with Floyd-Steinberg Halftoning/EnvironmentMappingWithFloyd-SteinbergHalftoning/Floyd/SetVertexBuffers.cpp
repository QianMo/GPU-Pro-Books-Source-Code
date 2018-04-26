#include "DXUT.h"
#include "SetVertexBuffers.h"
#include "ResourceLocator.h"
#include "Theatre.h"
#include "ScriptResourceVariable.h"
#include "TaskContext.h"

SetVertexBuffers::SetVertexBuffers()
{
}

void SetVertexBuffers::addBuffer(ScriptResourceVariable* buffer, unsigned int stride, unsigned int offset)
{
	buffers.push_back(buffer);
	strides.push_back(stride);
	offsets.push_back(offset);
}

void SetVertexBuffers::execute(const TaskContext& context)
{
	std::vector<ScriptResourceVariable*>::iterator i = buffers.begin();
	std::vector<unsigned int>::iterator iof = offsets.begin();
	std::vector<unsigned int>::iterator ist = strides.begin();
	unsigned int nTargets = 0;
	ID3D10Buffer* bufis[32];
	unsigned int ofis[32];
	unsigned int stricis[32];
	while(i != buffers.end())
	{
		ID3D10Resource* resi = (*i)->getResource();
		if(resi)
		{
			D3D10_RESOURCE_DIMENSION dim;
			resi->GetType(&dim);
			if(dim != D3D10_RESOURCE_DIMENSION_BUFFER)
				return;
		}
		bufis[nTargets] = (ID3D10Buffer*)resi;
		stricis[nTargets] = *ist;
		ofis[nTargets] = *iof;
		nTargets++;
		i++;
		ist++;
		iof++;
	}
	context.theatre->getDevice()->IASetVertexBuffers(0, nTargets, bufis, stricis, ofis);
}
