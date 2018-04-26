#include "DXUT.h"
#include "SetStreamTargets.h"
#include "ResourceLocator.h"
#include "Theatre.h"
#include "ScriptResourceVariable.h"
#include "TaskContext.h"

SetStreamTargets::SetStreamTargets()
{
}

void SetStreamTargets::addBuffer(ScriptResourceVariable* buffer, unsigned int offset)
{
	buffers.push_back(buffer);
	offsets.push_back(offset);
}

void SetStreamTargets::execute(const TaskContext& context)
{
	std::vector<ScriptResourceVariable*>::iterator i = buffers.begin();
	std::vector<unsigned int>::iterator iof = offsets.begin();
	unsigned int nTargets = 0;
	ID3D10Buffer* bufis[32];
	unsigned int ofis[32];
	while(i != buffers.end())
	{
		ID3D10Resource* resi;
		if(*i)
			resi = (*i)->getResource();
		else
			resi = NULL;
		if(resi)
		{
			D3D10_RESOURCE_DIMENSION dim;
			resi->GetType(&dim);
			if(dim != D3D10_RESOURCE_DIMENSION_BUFFER)
				return;
		}
		bufis[nTargets] = (ID3D10Buffer*)resi;
		ofis[nTargets] = *iof;
		nTargets++;
		i++;
		iof++;
	}
	context.theatre->getDevice()->SOSetTargets(nTargets, bufis, ofis);
}
