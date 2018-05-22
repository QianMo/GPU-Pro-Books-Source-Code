#include <stdafx.h>
#include <FILE_MANAGER.h>

void FILE_MANAGER::Release()
{
	directories.Erase();
}

void FILE_MANAGER::AddDirectory(const char *directory)
{
	if(!directory)
		return;
	for(int i=0;i<directories.GetSize();i++)
	{
		if(strcmp(directories[i],directory)==0)
			return;
	}
	char newDirectory[DEMO_MAX_STRING];
	strcpy(newDirectory,directory);
	directories.AddElement(&newDirectory);
}

void FILE_MANAGER::RemoveDirectories()
{
	directories.Clear();
}

bool FILE_MANAGER::FilePathExists(const char *filePath) const
{
	if(!filePath)
		return false;
	return (GetFileAttributes(filePath)!=0xFFFFFFFF); 
}

bool FILE_MANAGER::SetWorkDirectory(const char *workDirectory) const
{
	if(!workDirectory)
		return false;
	return (_chdir(workDirectory)==0);
}

bool FILE_MANAGER::GetExeDirectory(char *exeDirectory) const
{
	if(!exeDirectory)
		return false;
	if(GetModuleFileName(NULL,exeDirectory,DEMO_MAX_FILEPATH)==0)
		return false;
	std::string directory = exeDirectory;
	directory = directory.substr(0,directory.find_last_of('\\'));
	strcpy(exeDirectory,directory.c_str());
	return true;
}

bool FILE_MANAGER::GetFilePath(const char *fileName,char *filePath) const
{
	if((!fileName)||(!filePath))
		return false;
	for(int i=0;i<directories.GetSize();i++)
	{
		strcpy(filePath,directories[i]);
		strcat(filePath,fileName);
		if(FilePathExists(filePath))
			return true;
	}  
	return false;
}

bool FILE_MANAGER::GetFileName(const char *filePath,char *fileName) const
{
	if((!filePath)||(!fileName))
	  return false;
	std::string filename = filePath;
	int indexA = filename.find_last_of("/");
	int indexB = filename.find_last_of("\\");
	int index = (indexA>indexB) ? indexA : indexB;
	if(index>-1)
		filename.erase(0,index+1);
	index = filename.find_last_of(".");
	if(index>-1)
		filename.erase(index,filename.length()-index);
	strcpy(fileName,filename.c_str());
	return true;
}