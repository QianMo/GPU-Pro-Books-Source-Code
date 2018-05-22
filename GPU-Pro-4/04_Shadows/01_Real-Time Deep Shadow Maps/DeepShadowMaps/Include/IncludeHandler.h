#pragma once

#include "Core.h"
#include "Effects.h"

class IncludeHandler : public CoreIncludeHandler
{
public:
	IncludeHandler(int currentEffect) { this->currentEffect = currentEffect; };

	// Note: Cannot handle unicode filenames, do not end the file with a \0!
	CoreResult Open(D3D10_INCLUDE_TYPE IncludeType, std::wstring fileName, void** outData, UINT* outNumBytes);
	// Data = the pointer returned in Open
	void Close(void *data);
protected:
	int currentEffect;
};