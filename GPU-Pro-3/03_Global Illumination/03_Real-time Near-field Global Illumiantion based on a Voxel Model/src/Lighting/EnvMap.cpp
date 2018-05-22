#include "EnvMap.h"

#define MAX_LINE 1024 

EnvMap::EnvMap(string fileName)
{
   mEnvMapPixels = 0;
   mRotationAngle = 0.0f;
   loadPFM(fileName);
}

// from SSDO Demo
// load hdr image in PFM format
bool EnvMap::loadPFM(string fileName)
{
   cout << "Loading PFM" << endl;

   // Clean up before loading
   if(mEnvMapPixels)
      free(mEnvMapPixels);
   if(glIsTexture(mEnvMapTextureId))
      glDeleteTextures(1, &mEnvMapTextureId);

   // init some variables 
   char imageformat[ MAX_LINE ]; 
   float f[1]; 

   // open the file handle  
   FILE* infile = fopen( fileName.c_str(), "rb" ); 
   cout << "File handle opened" << endl;

   if ( infile == NULL )
   { 
      printf("Error loading %s !\n",fileName.c_str()); 
      exit(-1); 
   } 

   // read the header  
   fscanf( infile," %s %d %d ", &imageformat, &mEnvMapWidth, &mEnvMapHeight ); 

   // set member variables 
   // assert( width > 0 && height > 0 ); 
   printf("Image format %s Width %d Height %d\n",imageformat, mEnvMapWidth, mEnvMapHeight ); 

   mEnvMapPixels = (float*) (malloc(mEnvMapWidth * mEnvMapHeight * 3 * sizeof(float))); 

   // go ahead with the data 
   fscanf( infile,"%f", &f[0] ); 
   fgetc( infile ); 

   float red, green, blue; 

   float *p = mEnvMapPixels; 
   mLMax = 0.0f;
   // read the values and store them 
   for ( int j = 0; j < mEnvMapHeight ; j++ )  { 
      for ( int i = 0; i < mEnvMapWidth ; i++ )  { 

         fread( f, 4, 1, infile ); 
         red = f[0]; 

         fread( f, 4, 1, infile ); 
         green = f[0]; 

         fread( f, 4, 1, infile ); 
         blue = f[0]; 

         *p++ = red; 
         *p++ = green; 
         *p++ = blue; 

         float L = (red + green + blue) / 3.0f; 
         if (L > mLMax) 
            mLMax = L; 
      } 
   } 
   printf("Loading Envmap finished\n"); 
   printf("Maximum luminance: %f\n", mLMax); 

   mInvGamma = 1.0f / 2.2f; 

   createTextureFromLoadedPFM();

   return true;

}

void EnvMap::createTextureFromLoadedPFM()
{
   // texture for blurred envmap
   glGenTextures (1, &mEnvMapTextureId);
   glBindTexture (GL_TEXTURE_2D, mEnvMapTextureId);
   glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB32F_ARB, mEnvMapWidth, mEnvMapHeight, 0, GL_RGB, GL_FLOAT, mEnvMapPixels);
   //glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   //glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

}

void EnvMap::rotate(float dx)
{
   mRotationAngle += dx;
   if(mRotationAngle > 2.0f * F_PI)
      mRotationAngle -= 2 * F_PI;
}