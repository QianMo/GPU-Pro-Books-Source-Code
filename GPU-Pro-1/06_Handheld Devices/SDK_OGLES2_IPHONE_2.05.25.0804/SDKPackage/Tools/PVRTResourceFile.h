/******************************************************************************

 @File         PVRTResourceFile.h

 @Title        PVRTResourceFile

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Simple resource file wrapper

******************************************************************************/
#ifndef _PVRTRESOURCEFILE_H_
#define _PVRTRESOURCEFILE_H_

#include <stdlib.h>
#include "PVRTString.h"

/*!***************************************************************************
 @Class CPVRTResourceFile
 @Brief Simple resource file wrapper
*****************************************************************************/
class CPVRTResourceFile
{
public:
	/*!***************************************************************************
	@Function			SetReadPath
	@Input				pszReadPath The path where you would like to read from
	@Description		Sets the read path
	*****************************************************************************/
	static void SetReadPath(const char* pszReadPath);

	/*!***************************************************************************
	@Function			GetReadPath
	@Returns			The currently set read path
	@Description		Returns the currently set read path
	*****************************************************************************/
	static CPVRTString GetReadPath();

	/*!***************************************************************************
	@Function			CPVRTResourceFile
	@Input				pszFilename Name of the file you would like to open
	@Description		Constructor
	*****************************************************************************/
	CPVRTResourceFile(const char* pszFilename);

	/*!***************************************************************************
	@Function			CPVRTResourceFile
	@Input				pData A pointer to the data you would like to use
	@Input				i32Size The size of the data
	@Description		Constructor
	*****************************************************************************/
	CPVRTResourceFile(const char* pData, size_t i32Size);

	/*!***************************************************************************
	@Function			~CPVRTResourceFile
	@Description		Destructor
	*****************************************************************************/
	virtual ~CPVRTResourceFile();

	/*!***************************************************************************
	@Function			IsOpen
	@Returns			true if the file is open
	@Description		Is the file open
	*****************************************************************************/
	bool IsOpen() const;

	/*!***************************************************************************
	@Function			IsMemoryFile
	@Returns			true if the file was opened from memory
	@Description		Was the file opened from memory
	*****************************************************************************/
	bool IsMemoryFile() const;

	/*!***************************************************************************
	@Function			Size
	@Returns			The size of the opened file
	@Description		Returns the size of the opened file
	*****************************************************************************/
	size_t Size() const;

	/*!***************************************************************************
	@Function			DataPtr
	@Returns			A pointer to the file data
	@Description		Returns a pointer to the file data
	*****************************************************************************/
	const void* DataPtr() const;

	/*!***************************************************************************
	@Function			StringPtr
	@Returns			The file data as a string
	@Description		Returns the file as a null-terminated string
	*****************************************************************************/
	// convenience getter. Also makes it clear that you get a null-terminated buffer.
	const char* StringPtr() const;

	/*!***************************************************************************
	@Function			Close
	@Description		Closes the file
	*****************************************************************************/
	void Close();

protected:
	bool m_bOpen;
	bool m_bMemoryFile;
	size_t m_Size;
	const char* m_pData;

	static CPVRTString s_ReadPath;
};

#endif // _PVRTRESOURCEFILE_H_

/*****************************************************************************
 End of file (PVRTResourceFile.h)
*****************************************************************************/
