#include "IncludeHandler.h"
#include "MemoryLeakTracker.h"
#include <fstream>

CoreResult IncludeHandler::Open(D3D10_INCLUDE_TYPE IncludeType, std::wstring fileName, void** outData, UINT* outNumBytes)
{
	std::fstream in;
	in.open((EffectPaths[currentEffect] + L"\\" + fileName).c_str(), std::ios::in);

	if(!in.good())
	{
		CoreLog::Information(L"Could not open include " + fileName);
		return CORE_INVALIDARGS;
	}

	std::string text;
	std::string line;
	
	while (!in.eof())
    {
      getline (in, line);
      text += line + "\n";
    }

	in.close();
	*outNumBytes = (UINT)text.length();
	*outData = (void *)new char[*outNumBytes];
	if(!*outData)
		return CORE_OUTOFMEM;

	memcpy(*outData, text.c_str(), *outNumBytes);
	
	return CORE_OK;
}

void IncludeHandler::Close(void *data)
{
	SAFE_DELETE(data);
}