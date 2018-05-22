#include "Core.h"

void ICoreBase::Release()
{
	if(refCounter > 1)
		refCounter--;
	else
	{
		finalRelease();
		delete this;
	}
}