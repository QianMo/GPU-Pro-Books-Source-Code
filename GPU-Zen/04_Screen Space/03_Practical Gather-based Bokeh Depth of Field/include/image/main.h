#pragma once


#include "types.h"
#include "image.h"
#include "fourier.h"


namespace NImage
{
	void Initialize();
	void Deinitialize();

	void SetFreeImageCustomOutputMessageFunction(void (*_freeImageCustomOutputMessageFunction)(const string& msg));

	//

	extern void (*freeImageCustomOutputMessageFunction)(const string& msg);
}
