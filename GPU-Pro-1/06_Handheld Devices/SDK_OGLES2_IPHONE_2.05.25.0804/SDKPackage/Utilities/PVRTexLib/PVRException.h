/******************************************************************************

 @File         PVRException.h

 @Title        

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI

 @Description  Exception class and macros.

******************************************************************************/
#ifndef _PVREXCEPTION_H_
#define _PVREXCEPTION_H_

#ifndef PVR_DLL
#ifdef _WINDLL_EXPORT
#define PVR_DLL __declspec(dllexport)
#elif _WINDLL_IMPORT
#define PVR_DLL __declspec(dllimport)
#else
#define PVR_DLL
#endif
#endif

/*****************************************************************************
* Exception class and macros
* Use char* literals only for m_what.
*****************************************************************************/
class PVR_DLL PVRException
{
public:
	PVRException(const char* const what)throw();
	const char * const what() const;
	~PVRException() throw();
private:
	const char* const m_what;
};

#define PVRTRY			try
#define PVRLOGTHROW(A)	{PVRException myException(A); PVRTextureUtilities::getPointer()->log(A) ; throw(myException);}
#define PVRLOG			PVRTextureUtilities::getPointer()->log
#define PVRTHROW(A)		{PVRException myException(A); throw(myException);}
#define PVRCATCH(A)		catch(PVRException& A)
#define PVRCATCHALL		catch(...)

#endif // _PVREXCEPTION_H_

/*****************************************************************************
*  end of file
******************************************************************************/
