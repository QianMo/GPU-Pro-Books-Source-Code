#include <stdafx.h>
#include <FileManager.h>

void FileManager::AddDirectory(const char *directory)
{
  if(!directory)
    return;
  for(UINT i=0; i<directories.GetSize(); i++)
  {
    if(strcmp(directories[i], directory) == 0)
      return;
  }
  char newDirectory[DEMO_MAX_STRING];
  strcpy(newDirectory, directory);
  directories.AddElement(&newDirectory);
}

void FileManager::RemoveDirectories()
{
  directories.Clear();
}

bool FileManager::FilePathExists(const char *filePath) const
{
  if(!filePath)
    return false;

  return (GetFileAttributes(filePath) != 0xFFFFFFFF);
}

bool FileManager::SetWorkDirectory(const char *workDirectory) const
{
  if(!workDirectory)
    return false;
  return (_chdir(workDirectory) == 0);
}

bool FileManager::GetExeDirectory(char *exeDirectory) const
{
  if(!exeDirectory)
    return false;

  if(GetModuleFileName(nullptr, exeDirectory, DEMO_MAX_FILEPATH) == 0)
    return false;
  std::string directory = exeDirectory;
  directory = directory.substr(0, directory.find_last_of('\\'));
  strcpy(exeDirectory, directory.c_str());

  return true;
}

bool FileManager::GetFilePath(const char *fileName, char *filePath) const
{
  if((!fileName) || (!filePath))
    return false;
  for(UINT i=0; i<directories.GetSize(); i++)
  {
    strcpy(filePath, directories[i]);
    strcat(filePath, fileName);
    if(FilePathExists(filePath))
      return true;
  }  
  return false;
}

bool FileManager::GetFileName(const char *filePath, char *fileName) const
{
  if((!filePath) || (!fileName))
    return false;
  std::string filename = filePath;
  size_t indexA = filename.find_last_of("/");
  size_t indexB = filename.find_last_of("\\");
  size_t index = (indexA != std::string::npos) ? indexA : indexB;
  if(index != std::string::npos)
    filename.erase(0, index+1);
  index = filename.find_last_of(".");
  if(index > -1)
    filename.erase(index, filename.length()-index);
  strcpy(fileName, filename.c_str());
  return true;
}