#ifndef __FREP_LOADER_H__
#define __FREP_LOADER_H__

#include <fstream>
#include <sstream>
#include <strstream>

struct FREP_DESCRIPTION;

class FREP_LOADER 
{
public:

   static bool loadData(const std::string& modelName, FREP_DESCRIPTION* model);

private:
   // removes comments if needed
   static bool readLine(std::istrstream& stream, std::string* outputString);

   // read next available line (skipping comments)
   //bool readNextLine(std::ifstream& stream, std::istringstream* outputStream)
   static bool readNextLine(std::istrstream& stream, std::istringstream* outputStream);

   // read next available line (skipping comments)
   static std::istringstream& readNextLine(std::istrstream& stream);

};


#endif // __FREP_LOADER_H__