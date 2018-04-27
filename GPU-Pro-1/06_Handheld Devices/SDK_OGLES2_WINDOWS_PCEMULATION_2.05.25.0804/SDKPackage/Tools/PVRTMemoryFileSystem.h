/******************************************************************************

 @File         PVRTMemoryFileSystem.h

 @Title        PVRTMemoryFileSystem

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Memory file system for resource files

******************************************************************************/
#ifndef _PVRTMEMORYFILE_H_
#define _PVRTMEMORYFILE_H_

#include "PVRTGlobal.h"
#include <stddef.h>

class CPVRTMemoryFileSystem
{
public:
	CPVRTMemoryFileSystem(const char* pszFilename, const void* pBuffer, size_t Size, bool bCopy = false);

	/*!***************************************************************************
	 @Function		RegisterMemoryFile
	 @Input			pszFilename		Name of file to register
	 @Input			pBuffer			Pointer to file data
	 @Input			Size			File size
	 @Input			bCopy			Name and data should be copied?
	 @Description	Registers a block of memory as a file that can be looked up
	                by name.
	*****************************************************************************/
	static void RegisterMemoryFile(const char* pszFilename, const void* pBuffer, size_t Size, bool bCopy = false);

	/*!***************************************************************************
	 @Function		GetFile
	 @Input			pszFilename		Name of file to open
	 @Output		ppBuffer		Pointer to file data
	 @Output		pSize			File size
	 @Return		true if the file was found in memory, false otherwise
	 @Description	Looks up a file in the memory file system by name. Returns a
	                pointer to the file data as well as its size on success.
	*****************************************************************************/
	static bool GetFile(const char* pszFilename, const void** ppBuffer, size_t* pSize);

	//static std::string DebugOut();

	/*!***************************************************************************
	 @Function		GetNumFiles
	 @Return		The number of registered files
	 @Description	Getter for the number of registered files
	*****************************************************************************/
	static int GetNumFiles();

	/*!***************************************************************************
	 @Function		GetFilename
	 @Input			i32Index		Index of file
	 @Return		A pointer to the filename of the requested file
	 @Description	Looks up a file in the memory file system by name. Returns a
	                pointer to the file data as well as its size on success.
	*****************************************************************************/
	static const char* GetFilename(int i32Index);

protected:
	class CAtExit
	{
	public:
		/*!***************************************************************************
		@Function		Destructor
		@Description	Destructor of CAtExit class. Workaround for platforms that
		                don't support the atexit() function. This deletes any memory
						file system data.
		*****************************************************************************/
		~CAtExit();
	};
	static CAtExit s_AtExit;

	friend class CAtExit;

	struct SFileInfo
	{
		const char* pszFilename;
		const void* pBuffer;
		size_t Size;
		bool bAllocated;
	};
	static SFileInfo* s_pFileInfo;
	static int s_i32NumFiles;
	static int s_i32Capacity;
};

#endif // _PVRTMEMORYFILE_H_

/*****************************************************************************
 End of file (PVRTMemoryFileSystem.h)
*****************************************************************************/
