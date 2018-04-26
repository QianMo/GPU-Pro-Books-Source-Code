#pragma once

class XMLNode;

class FileFinder
{
	typedef std::map<std::wstring, std::wstring> AliasDirectory;
	typedef std::vector<std::wstring> FolderList;
	
	FolderList folders;
	AliasDirectory aliases;
public:
	FileFinder(XMLNode& playNode);

	void pushFolder(const std::wstring& folderPath);
	void popFolder();

	XMLNode openXMLFile(const std::wstring& fileName, const wchar_t* topLevelTag = NULL);
	std::wstring completeFileName(const std::wstring& fileName);
};
