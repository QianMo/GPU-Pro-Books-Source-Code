#include "DXUT.h"
#include "FileFinder.h"
#include "xmlParser.h"

FileFinder::FileFinder(XMLNode& playNode)
{
	pushFolder(L"./");
	int iUseFolder = 0;
	XMLNode useFolderNode;
	while( !(useFolderNode = playNode.getChildNode(L"useFolder", iUseFolder)).isEmpty() )
	{
		std::wstring folderPath = useFolderNode.readWString(L"path");
		pushFolder(folderPath);
		iUseFolder++;
	}
}

void FileFinder::pushFolder(const std::wstring& folderPath)
{
	std::wstring formattedFolderPath = folderPath;
	
	int lastPerPos = formattedFolderPath.find_last_of(L'/');
	if(lastPerPos != formattedFolderPath.size()-1)
	{
		formattedFolderPath.append(L"/");
	}

	WIN32_FILE_ATTRIBUTE_DATA info;
	int fileSucc = GetFileAttributesExW(
		formattedFolderPath.c_str(),
		GetFileExInfoStandard,
		(void*)&info);

	if(fileSucc)
	{
		if(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			folders.push_back(formattedFolderPath);
		else
			EggERR(L"Folder added to the path does not appear to be a folder: " << formattedFolderPath);
	}
	else
		EggERR(L"Folder added to the path appears to be non-existant: " << formattedFolderPath);
}

void FileFinder::popFolder()
{
	folders.pop_back();
}

XMLNode FileFinder::openXMLFile(const std::wstring& fileName, const wchar_t* topLevelTag)
{
/*	WIN32_FILE_ATTRIBUTE_DATA info;
	int fileSucc = GetFileAttributesExW(
		fileName.c_str(),
		GetFileExInfoStandard,
		(void*)&info);
	if(fileSucc)
	{
		tNode = XMLNode::openFileHelper(fileName.c_str(), topLevelTag);
		if(!tNode.isEmpty())
			return tNode;
	}*/
	XMLNode tNode;
	FolderList::reverse_iterator iFolder = folders.rbegin();
	FolderList::reverse_iterator eFolder = folders.rend();
	while(iFolder != eFolder)
	{
		std::wstring combinedName = *iFolder + fileName;
		WIN32_FILE_ATTRIBUTE_DATA info;
		int fileSucc = GetFileAttributesExW(
			combinedName.c_str(),
			GetFileExInfoStandard,
			(void*)&info);
		if(fileSucc)
		{
			tNode = XMLNode::openFileHelper(combinedName.c_str(), topLevelTag);
/*			int lastPerPos = combinedName.find_last_of(L'/');
			if(lastPerPos > iFolder->size())
			{
				std::wstring fPath = combinedName.substr(0, lastPerPos);
				pushFolder(fPath);
			}*/
			if(!tNode.isEmpty())
				return tNode;
		}
		iFolder++;
	}
	EggERR(L"XML file could not be found: " << fileName);
	return tNode;	
}

std::wstring FileFinder::completeFileName(const std::wstring& fileName)
{
/*	WIN32_FILE_ATTRIBUTE_DATA info;
	int fileSucc = GetFileAttributesExW(
		fileName.c_str(),
		GetFileExInfoStandard,
		(void*)&info);
	if(fileSucc)
	{
		return fileName;
	}*/
	FolderList::reverse_iterator iFolder = folders.rbegin();
	FolderList::reverse_iterator eFolder = folders.rend();
	while(iFolder != eFolder)
	{
		std::wstring combinedName = *iFolder + fileName;
		WIN32_FILE_ATTRIBUTE_DATA info;
		int fileSucc = GetFileAttributesExW(
			combinedName.c_str(),
			GetFileExInfoStandard,
			(void*)&info);
		if(fileSucc)
		{
			return combinedName;
		}
		iFolder++;
	}
	EggERR(L"File could not be found: " << fileName);
	return fileName;	
}