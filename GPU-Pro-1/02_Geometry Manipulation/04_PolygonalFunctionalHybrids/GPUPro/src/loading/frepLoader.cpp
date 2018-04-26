
#include <fstream>
#include <sstream>
#include <strstream>

#include <assert.h>

#include "../struct.h"

#include "frepLoader.h"

// removes comments if needed
bool FREP_LOADER::readLine(std::istrstream& stream, std::string* outputString)
{
   if (outputString == NULL) {
      assert(0);
      return false;
   }

   const int MAX_BUFFER_LENGTH = 256;
   char tmpBuf[MAX_BUFFER_LENGTH];
   
   stream.getline(tmpBuf, MAX_BUFFER_LENGTH);

   std::string line(tmpBuf);

   if (line.empty() || line.find('#') != std::string::npos) {
      *outputString = "";
      return false;
   }

   *outputString = tmpBuf;

   return true;
}

// read next available line (skipping comments)
//bool readNextLine(std::ifstream& stream, std::istringstream* outputStream)
bool FREP_LOADER::readNextLine(std::istrstream& stream, std::istringstream* outputStream)
{
   if (outputStream == NULL) {
      assert(0);
      return false;
   }

   std::string outputString;

   bool lineWasRead = false;

   while (stream.good() && !lineWasRead) {
      lineWasRead = readLine(stream, &outputString);
   }
   
   outputStream->str(outputString);

   return true;
}

// read next available line (skipping comments)
std::istringstream& FREP_LOADER::readNextLine(std::istrstream& stream)
//std::istringstream readNextLine(std::ifstream& stream)
{ 

   std::string outputString;

   bool lineWasRead = false;

   while (stream.good() && !lineWasRead) {
      lineWasRead = readLine(stream, &outputString);
   }

   static std::istringstream outputStream;

   outputStream.clear();
   outputStream.str(outputString);   

   return outputStream;
}


bool FREP_LOADER::loadData(const std::string& modelName, FREP_DESCRIPTION* model)
{

   std::ifstream fileStream(modelName.c_str());

   if (!fileStream.good() || model == NULL) {
		assert(0);
		return false;
	}
  
   std::istreambuf_iterator<char> fileStreamIterator( fileStream );
   std::istreambuf_iterator<char> endOfStream;

   std::string str( fileStreamIterator, endOfStream );   
   std::istrstream stream(str.c_str());
   
   BOUNDING_BOX& volumeBox = model->polygonizationParams.volumeBox;

   readNextLine(stream) >> volumeBox.minX >> volumeBox.minY >> volumeBox.minZ 
                        >> volumeBox.maxX >> volumeBox.maxY >> volumeBox.maxZ;
   
   readNextLine(stream) >> model->isBlendingOn;

   readNextLine(stream) >> model->convolutionIsoValue;

   int segmentsNum(0);
   readNextLine(stream) >> segmentsNum;
   
   int framesNum(0);
   readNextLine(stream) >> framesNum;

   model->sampledModel.resize(framesNum);

   // read model parameters sampled at each frame
   for (int i = 0; i < framesNum; i++) {

      MODEL_PARAMETERS& parameters = model->sampledModel[i];

      if (model->isBlendingOn) {
         
         readNextLine(stream) >>   parameters.submodelParams.blendingParams.a0 >> 
                                       parameters.submodelParams.blendingParams.a1 >> parameters.submodelParams.blendingParams.a2;         

         readNextLine(stream) >>   parameters.submodelParams.implicitBox.minX >> parameters.submodelParams.implicitBox.minY >> parameters.submodelParams.implicitBox.minZ >> 
                                       parameters.submodelParams.implicitBox.maxX >> parameters.submodelParams.implicitBox.maxY >> parameters.submodelParams.implicitBox.maxZ;
      }

      parameters.segments.resize(segmentsNum);
      // now read parameters of all line segments
      for (int segmentIdx = 0; segmentIdx < segmentsNum; segmentIdx++) {

         CONVOLUTION_SEGMENT& segment = parameters.segments[segmentIdx];

         readNextLine(stream) >> segment.x1 >> segment.y1 >> segment.z1 >> segment.x2 >> segment.y2 >> segment.z2 >> segment.s;
      }     
      
   }

   fileStream.close();

   return true;
}
