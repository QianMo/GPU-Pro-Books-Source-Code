#include <stdafx.h>
#include <FileManager.h>

void FileManager::AddDirectory(const char *directory)
{
  if(!directory)
    return;
  for(unsigned int i=0; i<directories.GetSize(); i++)
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

#ifndef UNIX_PORT
  return (GetFileAttributes(filePath) != 0xFFFFFFFF);
#else
  struct stat st;
  if(stat(filePath, &st) != 0)
  {
    return false;
  }
  auto f = st.st_mode & S_IFMT;
  return f == S_IFREG || f == S_IFDIR;
#endif /* UNIX_PORT */
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

#ifndef UNIX_PORT
  if(GetModuleFileName(NULL, exeDirectory, DEMO_MAX_FILEPATH) == 0)
    return false;
  std::string directory = exeDirectory;
  directory = directory.substr(0, directory.find_last_of('\\'));
  strcpy(exeDirectory, directory.c_str());
#else
  exeDirectory[0] = '.';
  exeDirectory[1] = 0;
#endif /* UNIX_PORT */

  return true;
}

bool FileManager::GetFilePath(const char *fileName, char *filePath) const
{
  if((!fileName) || (!filePath))
    return false;
  for(unsigned int i=0; i<directories.GetSize(); i++)
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
  int indexA = filename.find_last_of("/");
  int indexB = filename.find_last_of("\\");
  int index = (indexA > indexB) ? indexA : indexB;
  if(index > -1)
    filename.erase(0, index+1);
  index = filename.find_last_of(".");
  if(index > -1)
    filename.erase(index, filename.length()-index);
  strcpy(fileName, filename.c_str());
  return true;
}