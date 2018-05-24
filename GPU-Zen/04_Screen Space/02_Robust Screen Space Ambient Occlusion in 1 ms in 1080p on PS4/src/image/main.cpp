#include <image/main.h>
#include <essentials/main.h>

#include <FreeImage.h>


// utils


void FreeImageCustomOutputMessageFunction(FREE_IMAGE_FORMAT fif, const char* msg)
{
	SAFE_CALL(NImage::freeImageCustomOutputMessageFunction)(string("ERROR (FreeImage): ") + string(msg));
}


//


void NImage::Initialize()
{
	FreeImage_Initialise();
	FreeImage_SetOutputMessage(FreeImageCustomOutputMessageFunction);
}


void NImage::Deinitialize()
{
	FreeImage_DeInitialise();
}


void NImage::SetFreeImageCustomOutputMessageFunction(void (*_freeImageCustomOutputMessageFunction)(const string& msg))
{
	freeImageCustomOutputMessageFunction = _freeImageCustomOutputMessageFunction;
}


//


void (*NImage::freeImageCustomOutputMessageFunction)(const string& msg);
