#ifndef FILE_MANAGER_H
#define FILE_MANAGER_H

#include <List.h>

// FileManager
//
// Simple manager for file operations.
class FileManager
{
public:
  // adds new search directory for data
  void AddDirectory(const char *directory);

  // removes all search directories
  void RemoveDirectories();

  // checks, if file path exists
  bool FilePathExists(const char *filePath) const;

  // sets working directory
  bool SetWorkDirectory(const char *workDirectory) const;

  // gets directory, in which executable is located
  bool GetExeDirectory(char *exeDirectory) const;

  // gets file path for specified file name
  bool GetFilePath(const char *fileName, char *filePath) const;

  // gets file name for specified file path
  bool GetFileName(const char *filePath, char *fileName) const;

private:
  List<char[DEMO_MAX_STRING]> directories;
  
};

#endif