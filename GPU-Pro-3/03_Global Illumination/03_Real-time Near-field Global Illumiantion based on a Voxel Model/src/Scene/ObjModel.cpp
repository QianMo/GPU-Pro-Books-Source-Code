#include "ObjModel.h"

#include "Utils/GLError.h"
#include "Utils/ShaderProgram.h"
#include "Utils/TexturePool.h"

void ObjModel::initSettings()
{
   mComputedVertexNormals = false;
   mUnitized = false;
   mCentered = false;
   mScaleFactor = 1.0;
   mVertexNormalsAngle = 0.0;
   mVertexNormalsSmoothingGroups = false;
}

/// \param file Path to obj-model file
/// \param usemtlGroups Indicates whether to group materials by usemtl-lines
/// or to group by g-lines (uses last usemtl as material for active group g)
///

ObjModel::ObjModel(string file, bool usemtlGroups) :
mUsemtlGroups(usemtlGroups)
{
   mVertices     = new vector<GLMvector>;
   mNormals      = new vector<GLMvector> ;      
   mTexCoords    = new vector<GLMtexCoord>;  
   mFacetNorms   = new vector<GLMvector>;   
   mTriangles    = new vector<GLMtriangle>;  
   mMaterials    = new vector<GLMmaterial>; 
   mGroups       = new vector<GLMgroup>;       
   mSmoothGroups = new vector<GLMsmoothGroup>;
   mTextures     = new vector<GLMtexture>; 
   mAtlasTexCoords = 0;
   mOwnVectorData = true;
   mCopiedTriangles = false;
   mPathAtlas = "";

   initSettings();

   glmReadOBJ(file);
}

ObjModel::ObjModel(string file, const ObjModel* refModel)
{
   mOwnVectorData = false;
   mVertices     = new vector<GLMvector>;
   mFacetNorms   = new vector<GLMvector>;   
   mNormals      = new vector<GLMvector>;  

   this->mTriangles = refModel->mTriangles;
   this->mMaterials = refModel->mMaterials;
   this->mTextures  = refModel->mTextures;
   this->mTexCoords = refModel->mTexCoords;
   this->mGroups    = refModel->mGroups;
   this->mSmoothGroups = refModel->mSmoothGroups;
   this->mAtlasTexCoords = refModel->mAtlasTexCoords;

   this->mMtlLibName = refModel->mMtlLibName;
   this->mUsemtlGroups = refModel->mUsemtlGroups;


   mCopiedTriangles = false;
   mPathAtlas = "[shared]" + refModel->getPathAtlas();

   initSettings();

   glmReadOBJVertices(file);
}

ObjModel::~ObjModel()
{
   deleteDisplayLists();

   if(mOwnVectorData)
   {
      deleteTextures();
      delete mTexCoords;
      delete mAtlasTexCoords;
      delete mTriangles;
      delete mMaterials;
      delete mGroups;
      delete mSmoothGroups;
      delete mTextures;
   }

   if(!mOwnVectorData && mCopiedTriangles)
   {
      delete mTriangles;
   }

   delete mVertices;
   delete mNormals;
   delete mFacetNorms;
}

GLvoid ObjModel::deleteTextures()
{
   for(unsigned int i = 0; i < mTextures->size(); i++)
   {
      if(glIsTexture(mTextures->at(i).id))
         glDeleteTextures(1, &mTextures->at(i).id);
   }
   mTextures->clear();
}


GLvoid ObjModel::deleteDisplayLists()
{
   for(map<GLuint, GLuint>::iterator it = mDisplayLists.begin(); it != mDisplayLists.end(); it++)
   {
      if(glIsList((*it).second))
         glDeleteLists((*it).second, 1);
   }
   mDisplayLists.clear();
}


GLvoid ObjModel::glmReadOBJVertices(string file)
{
   /* open file */
   ifstream inFile(file.c_str());
   if(!inFile)
   {
      cerr << "[ObjModel] glmReadOBJVertices() failed: can't open data file " << file << endl;
      return;
   }

   //cout << endl;
   //cout << "--------------------------------" << endl;
   cout << "OBJ FILE LOADING (ONLY VERTICES) "  << file << "\r";//<< endl;
   //cout << "--------------------------------" << endl << endl;

   // model data
   mPathModel     = file;
   mPosition[0]   = 0.0;
   mPosition[1]   = 0.0;
   mPosition[2]   = 0.0;

   // vertex count etc starts with 1, so push 1 dummy back:
   GLMvector dummyVector;
   mVertices->push_back(dummyVector);
   mNormals->push_back(dummyVector);

   string line;
   stringstream lineStream;
   string token;
   vector<string> tokens;

   while (getline(inFile, line)) 
   {
      // A line was read successfully.
      // Tokenize the line using a string stream.

      lineStream.clear();
      lineStream.str(line);
      tokens.clear();

      //get all tokens
      while(lineStream >> token)
      {
         tokens.push_back(token);
      }


      // if there are tokens
      if(!tokens.empty())
      {
         //Process tokens

         // v: vertex
         if(!tokens.at(0).compare("v"))
         {
            //cout << "v detected" << endl;
            assert(tokens.size()>=4);
            mVertices->push_back(convertToVector(tokens.at(1),tokens.at(2),tokens.at(3)));

            // debug output
            //cout <<	mVertices->back().x << " " << mVertices->back().y << " " << mVertices->back().z << endl;
         }
         // vn: vertex normal
         if(!tokens.at(0).compare("vn"))
         {
            //cout << "vn detected" << endl;
            assert(tokens.size()>=4);
            mNormals->push_back(convertToVector(tokens.at(1),tokens.at(2),tokens.at(3)));

            // debug output
            //cout <<	mNormals->back().x << " " << mNormals->back().y << " " << mNormals->back().z << endl;
         }
      }
   }
   inFile.close();
}

GLvoid ObjModel::glmReadOBJ(string file)
{
   /* open file */
   ifstream inFile(file.c_str());
   if(!inFile)
   {
      cerr << "[ObjModel] glmReadOBJ() failed: can't open data file " << file << endl;
      return;
   }

   cout << endl;
   cout << "----------------" << endl;
   cout << "OBJ FILE LOADING: "  << file << endl;
   cout << "----------------" << endl << endl;

   // model data
   mPathModel      = file;
   mPosition[0]   = 0.0;
   mPosition[1]   = 0.0;
   mPosition[2]   = 0.0;

   // vertex count etc starts with 1, so push 1 dummy back:
   GLMvector dummyVector;
   GLMtexCoord dummyTexCoord;
   GLMtriangle dummyTriangle;
   mVertices->push_back(dummyVector);
   mNormals->push_back(dummyVector);
   mTexCoords->push_back(dummyTexCoord);
   mTriangles->push_back(dummyTriangle);


   /* make a default group */
   /*  obj spec: The default group name is default. */
   unsigned int currentGroupIndex = glmFindOrAddGroup("default");
   assert(mGroups->size()>0);
   unsigned int currentMaterialIndex = 0; // the default material
   unsigned int currentSmoothingGroupIndex = glmFindOrAddSmoothGroup(0); // 0: smooth off

   // set a default material
   GLMmaterial defaultMaterial;
   defaultMaterial.name = "";
   defaultMaterial.shininess = 0.0f;
   defaultMaterial.id = 1;
   defaultMaterial.diffuse[0] = 0.8f;
   defaultMaterial.diffuse[1] = 0.8f;
   defaultMaterial.diffuse[2] = 0.8f;
   defaultMaterial.diffuse[3] = 1.0f;
   defaultMaterial.ambient[0] = 0.2f;
   defaultMaterial.ambient[1] = 0.2f;
   defaultMaterial.ambient[2] = 0.2f;
   defaultMaterial.ambient[3] = 1.0f;
   defaultMaterial.specular[0] = 0.0f;
   defaultMaterial.specular[1] = 0.0f;
   defaultMaterial.specular[2] = 0.0f;
   defaultMaterial.specular[3] = 1.0f;

   defaultMaterial.has_map_Kd = false;
   defaultMaterial.index_map_Kd = 0;

   defaultMaterial.senderScaleFactor = 1.0f;

   // hold a default material as first entry in mMaterials vector
   mMaterials->push_back(defaultMaterial);


   // first: find "mtllib" in file and read it 
   string line;
   stringstream lineStream;
   string token;
   vector<string> tokens;

   /* read in file */
   while (getline(inFile, line)) 
   {
      // A line was read successfully
      // Tokenize the line using a string stream.

      lineStream.clear();
      lineStream.str(line);
      tokens.clear();

      //get all tokens
      while(lineStream >> token)
      {
         tokens.push_back(token);
      }

      // if there are tokens
      if(!tokens.empty())
      {
         //Process tokens

         // mtllib: material library name
         if(!tokens.at(0).compare("mtllib"))
         {
            assert(tokens.size()>1);
            mMtlLibName = tokens.at(1);
            cout << "[ObjModel] mtllib detected: " << mMtlLibName << endl;
            glmReadMTL(mMtlLibName);
            break;
         }
      }
   }

   /* read in file */
   inFile.clear();              // forget we hit the end of file
   inFile.seekg(0, ios::beg);   // move to the start of the file
   stringstream strStream;

   while (getline(inFile, line)) 
   {
      // Tokenize line

      lineStream.clear();
      lineStream.str(line);
      tokens.clear();

      while(lineStream >> token)
      {
         tokens.push_back(token);
      }


      // if there are tokens
      if(!tokens.empty())
      {
         //Process tokens

         // v: vertex
         if(!tokens.at(0).compare("v"))
         {
            //cout << "v detected" << endl;
            assert(tokens.size()>=4);
            mVertices->push_back(convertToVector(tokens.at(1),tokens.at(2),tokens.at(3)));

            // debug output
            //cout <<	mVertices->back().x << " " << mVertices->back().y << " " << mVertices->back().z << endl;
         }
         // vt: vertex texture coordinate
         else if(!tokens.at(0).compare("vt"))
         {
            //cout << "vt detected" << endl;
            assert(tokens.size()>=3);
            mTexCoords->push_back(convertToTexCoord(tokens.at(1),tokens.at(2)));

            // debug output
            //cout <<	mTexCoords.back().s << " " << mTexCoords.back().t << endl;
         }
         // vn: vertex normal
         else if(!tokens.at(0).compare("vn"))
         {
            //cout << "vn detected" << endl;
            assert(tokens.size()>=4);
            mNormals->push_back(convertToVector(tokens.at(1),tokens.at(2),tokens.at(3)));

            // debug output
            //cout <<	mNormals->back().x << " " << mNormals->back().y << " " << mNormals->back().z << endl;
         }
         // f: face
         else if(!tokens.at(0).compare("f"))
         {
            //	cout << endl << "f detected" << endl;

            // for a triangle there must be 3 further tokens
            assert(tokens.size()>3);
            /* possible:
            f v       v       v
            f v/vt    v/vt    v/vt
            f v//vn   v//vn   v//vn 
            f v/vt/vn v/vt/vn v/vt/vn
            */

            // create triangles from polygons with more than 3 vertices:
            // number of triangles = num(tokens after "f") - 2
            int numTriangles = tokens.size()-1-2;

            for(int t=0; t<numTriangles; t++)
            {
               // Make an empty triangle
               GLMtriangle triangle;
               triangle.fnIndex = 0;
               triangle.smooth = (currentSmoothingGroupIndex!=0);
               unsigned int tokenIndices[] = {1, 2+t, 3+t};

               for(int vertex=1; vertex<=3; vertex++)
               {
                  // init triangle with 0 references 
                  // (a valid reference is: >= 1)
                  triangle.vIndices[vertex-1] = 0;
                  triangle.vtIndices[vertex-1] = 0;
                  triangle.vatIndices[vertex-1] = 0;
                  triangle.vnIndices[vertex-1] = 0;

                  string vertexToken;
                  strStream.clear();
                  strStream.str(tokens.at(tokenIndices[vertex-1]));

                  //cout << "processing : " << tokens.at(vertex) << endl;
                  int count = 0;
                  while ( getline(strStream, vertexToken, '/') )
                  {
                     if(!vertexToken.empty())
                     {
                        switch(count)
                        {
                        case 0:
                           triangle.vIndices[vertex-1] = atoi(vertexToken.c_str());
                           break;
                        case 1:
                           triangle.vtIndices[vertex-1] = atoi(vertexToken.c_str());
                           break;
                        case 2:
                           triangle.vnIndices[vertex-1] = atoi(vertexToken.c_str());
                           break;
                        default: break;
                        }
                     }
                     count++;
                  }
                  //std::cout << "v: " << triangle.vIndices[vertex-1] << std::endl;
                  //std::cout << "vt: " << triangle.vtIndices[vertex-1] << std::endl;
                  //std::cout << "vn: " << triangle.vnIndices[vertex-1] << std::endl;

               }
               // done with triangle

               // add triangle to mTriangles vector
               mTriangles->push_back(triangle);

               unsigned int triangleIndex = mTriangles->size()-1;
               // add index of current triangle to current group
               mGroups->at(currentGroupIndex).triangles.push_back(triangleIndex);

               // add index of current triangle to current smooth group
               mSmoothGroups->at(currentSmoothingGroupIndex).triangles.push_back(triangleIndex);
            }

         }// f
         // g: group
         else if(!tokens.at(0).compare("g"))
         {
            if(!mUsemtlGroups)
            {
               // process g group_name
               if(tokens.size() > 1)
               {
                  // group name is specified
                  currentGroupIndex = glmFindOrAddGroup(tokens.at(1));
               }
               else
               {
                  // use group with empty name
                  currentGroupIndex = glmFindOrAddGroup("");
               }
               // set the current material as this group's material
               mGroups->at(currentGroupIndex).materialIndex = currentMaterialIndex;
            }
         }
         // usemtl: material name
         else if(!tokens.at(0).compare("usemtl"))
         {
            assert(tokens.size()>1);
            //cout << "usemtl detected "<<tokens.at(1) << endl;
            currentMaterialIndex = glmFindMaterial(tokens.at(1));

            if(mUsemtlGroups)
            {
               // create a group for this material
               currentGroupIndex = glmFindOrAddGroup("group_"+tokens.at(1));
            }

            mGroups->at(currentGroupIndex).materialIndex = currentMaterialIndex;

         }
         // s: smoothing group
         else if(!tokens.at(0).compare("s"))
         {
            assert(tokens.size()>1);
            unsigned int smoothingGroupID = 0;
            // if token != "off"
            if(tokens.at(1).compare("off"))
               smoothingGroupID = atoi(tokens.at(1).c_str());
            currentSmoothingGroupIndex = glmFindOrAddSmoothGroup(smoothingGroupID);
         }

      }//tokens

      //new line;
   }
   inFile.close();

   cout << endl;
   cout << "[ObjModel] Created " << mGroups->size() << " material groups" << endl;
   cout << "[ObjModel] Created " << mSmoothGroups->size() << " smoothing groups" << endl;
   cout << endl;

   // check smoothing groups
   int num = 0;
   for(unsigned int i = 0; i < mSmoothGroups->size(); i++)
   {
      num += mSmoothGroups->at(i).triangles.size();
   }
   //cout << "[ObjModel] " <<  mVertices->size()-1 /* dummy vertex */ << " vertices in model." << endl;
   cout << "[ObjModel] " <<  mTriangles->size()-1 /* dummy triangle */ << " triangles in model." << endl;

   //cout << "[ObjModel] glmReadOBJ: Done" << endl;

}

GLvoid ObjModel::addAtlasTextureCoordinates(string file)
{
   mAtlasTexCoords = new vector<GLMtexCoord>;
   mAtlasTexCoords->push_back(GLMtexCoord(0,0));

   
   /* open file */
   ifstream inFile(file.c_str());
   if(!inFile)
   {
      cerr << "addAtlasTextureCoordinates() failed: can't open data file " << file << endl;
      return;
   }

   mPathAtlas = file;

   cout << "[ObjModel] Adding atlas texture coordinates: " << file << " ... ";

   string line, token;
   stringstream lineStream;
   vector<string> tokens;

   int currentFace = 0; // no face read in yet
   stringstream strStream;
   string vertexToken;


   /* read in file */
   while (getline(inFile, line)) 
   {
      // A line was read successfully, so you can process it
      // If you then want to tokenize the line use a string stream:

      lineStream.clear();
      lineStream.str(line);
      tokens.clear();

      //get all tokens
      while(lineStream >> token)
      {
         tokens.push_back(token);
      }

      // if there are tokens
      if(!tokens.empty())
      {
         //Process tokens

         // vt: vertex atlas texture coordinate
         if(!tokens.at(0).compare("vt"))
         {
            assert(tokens.size()>=3);
            mAtlasTexCoords->push_back(convertToTexCoord(tokens.at(1),tokens.at(2)));

            // debug output
            //cout <<	mAtlasTexCoords->size() << " " << mAtlasTexCoords->back().s << " " << mAtlasTexCoords->back().t << endl;

         }

         // f: face
         else if(!tokens.at(0).compare("f"))
         {
            //	cout << endl << "f detected" << endl;

            // for a triangle there must be 3 further tokens
            assert(tokens.size()>3);
           
            /* looking for:
            f v/vt    v/vt    v/vt
            */

            // create triangles from polygons with more than 3 vertices:
            // number of triangles = num(tokens after "f") - 2
            int numTriangles = tokens.size()-1-2;

            // detected numTriangles new faces

            for(int t=0; t<numTriangles; t++)
            {
               currentFace++; 
               // access the current face in the following with
               //mTriangles->at(currentFace)

               unsigned int tokenIndices[] = {1, 2+t, 3+t};

               for(int vertex=1; vertex<=3; vertex++)
               {
                  strStream.clear();
                  strStream.str(tokens.at(tokenIndices[vertex-1]));

                  //cout << "processing : " << tokens.at(vertex) << endl;
                  int count = 0;
                  while ( getline(strStream, vertexToken, '/') )
                  {
                     if(!vertexToken.empty())
                     {
                        if(count == 1) // vt deteced
                        {
                           if(mTriangles->size()>= unsigned int(currentFace+1))
                              mTriangles->at(currentFace).vatIndices[vertex-1] = atoi(vertexToken.c_str());
                        }
                     }
                     count++;
                  }
               }
               // done with triangle

            } // end triangles

         }// end f
      }
   }

   std::cout << "done" << std::endl;
}


string ObjModel::glmDirName(string path)
{
   size_t found;
   //cout << "[ObjModel] glmDirName: Splitting: " << path << endl;
   found = path.find_last_of("/\\");
   string folder = (found!=string::npos)?path.substr(0, found) : "";
   //cout << "[ObjModel] glmDirName: folder: " << folder << endl;
   //cout << "[ObjModel] glmDirName: file: " << path.substr(found+1) << endl;

   return folder;
}



GLvoid ObjModel::glmReadMTL(string file)
{
   string dirObj = glmDirName(mPathModel);
   string dirMtl = glmDirName(file);
   string mtlfile = dirObj+"/"+file;

   cout << "[ObjModel] glmReadMTL: Reading mtllib " << mtlfile << endl;

   ifstream inFile(mtlfile.c_str());
   if(!inFile)
   {
      cerr << "[ObjModel] glmReadMTL() failed: can't open material file " << mtlfile << endl;
   }

   GLMmaterial defaultMaterial = mMaterials->at(0);

   /* now, read in the data */

   string line;
   while (getline(inFile, line)) 
   {
      stringstream lineStream(line);
      string token;
      vector<string> tokens;

      //get all tokens
      while(lineStream >> token)
      {
         tokens.push_back(token);
      }

      // if there are tokens
      if(!tokens.empty())
      {
         //Process tokens

         if(!tokens.at(0).compare("newmtl"))
         {
            assert(tokens.size()>1);
            mMaterials->push_back(defaultMaterial);
            mMaterials->back().name = tokens.at(1);
         }
         // cheating factor for rendering sender objects darker
         else if(!tokens.at(0).compare("senderMatScale"))
         {
            assert(tokens.size()>1);
            mMaterials->back().senderScaleFactor = GLfloat(atof(tokens.at(1).c_str()));
            mMaterials->back().diffuse[3] = mMaterials->back().senderScaleFactor;

         }

         // ambient reflectivity
         else if(!tokens.at(0).compare("Ka"))
         {
            assert(tokens.size()>3);
            mMaterials->back().ambient[0] = GLfloat(atof(tokens.at(1).c_str()));
            mMaterials->back().ambient[1] = GLfloat(atof(tokens.at(2).c_str()));
            mMaterials->back().ambient[2] = GLfloat(atof(tokens.at(3).c_str()));
         }
         // diffuse reflectivity
         else if(!tokens.at(0).compare("Kd"))
         {
            assert(tokens.size()>3);
            mMaterials->back().diffuse[0] = GLfloat(atof(tokens.at(1).c_str()));
            mMaterials->back().diffuse[1] = GLfloat(atof(tokens.at(2).c_str()));
            mMaterials->back().diffuse[2] = GLfloat(atof(tokens.at(3).c_str()));
         }
         // specular reflectivity 
         else if(!tokens.at(0).compare("Ks"))
         {
            assert(tokens.size()>3);
            mMaterials->back().specular[0] = GLfloat(atof(tokens.at(1).c_str()));
            mMaterials->back().specular[1] = GLfloat(atof(tokens.at(2).c_str()));
            mMaterials->back().specular[2] = GLfloat(atof(tokens.at(3).c_str()));
         }
         // alpha
         else if(!tokens.at(0).compare("d") || !tokens.at(0).compare("Tr")) 
         {
            //cout << "glmReadMTL: alpha detected " << endl;
         }
         // diffuse texture map
         else if(!tokens.at(0).compare("map_Kd"))
         {
            assert(tokens.size()>1);
            mMaterials->back().index_map_Kd = glmFindOrAddTexture(dirObj+"/"+dirMtl+"/"+tokens.at(1), false);
            mMaterials->back().has_map_Kd = true;
            // set diffuse color to white because this material is textured
            //mMaterials->back().diffuse[0] = 1.0;
            //mMaterials->back().diffuse[1] = 1.0;
            //mMaterials->back().diffuse[2] = 1.0;
            //mMaterials->back().diffuse[3] = 1.0;
         }   
      }
   }
}




unsigned int ObjModel::glmFindOrAddGroup(const string& name)
{
   unsigned int index=0;

   // if did not find group name, add new one at the back
   if (!glmFindGroup(name, index)) 
   {
      //cout << "glmFindOrAddGroup: pushing group back: " << name << endl;

      GLMgroup group;
      group.name = name;
      group.materialIndex = 0; // the default material
      mGroups->push_back(group);

      // return the index of the last element in vector mGroups
      return mGroups->size()-1;
   }
   // found group
   else
   {
      // return its index
      return index;
   }

}



bool ObjModel::glmFindGroup(string name, unsigned int& index)
{
   // iterate through vector to find a group with name name
   for (unsigned int i = 0; i < mGroups->size(); i++)
   {
      if(!(mGroups->at(i)).name.compare(name))
      {
         //cout << "glmFindGroup: found group called " << name << endl;
         index = i;
         return true;
      }
   }
   //cout << "glmFindGroup: did not find group called " << name << endl;
   return false;
}



unsigned int ObjModel::glmFindOrAddSmoothGroup(unsigned int id)
{
   unsigned int index=0;
   // if did not find group name, add new one at the back
   if (!glmFindSmoothGroup(id, index)) 
   {
      //cout << "glmFindOrAddSmoothGroup: pushing group back: " << id << endl;

      GLMsmoothGroup s;
      s.id = id;
      mSmoothGroups->push_back(s);

      // return the index of the last element in vector mGroups
      return mSmoothGroups->size()-1;
   }
   // found group
   else
   {
      // return its index
      return index;
   }

}


bool ObjModel::glmFindSmoothGroup(unsigned int id, unsigned int& index)
{
   // iterate through vector to find a group with name name
   for (unsigned int i = 0; i < mSmoothGroups->size(); i++)
   {
      if(mSmoothGroups->at(i).id == id)
      {
         //cout << "glmFindSmoothGroup: found group with id " << id << endl;
         index = i;
         return true;
      }
   }
   //cout << "glmFindSmoothGroup: did not find group with id " << id << endl;
   return false;
}


GLMvector ObjModel::convertToVector(const string& token1, const string& token2, const string& token3)
{
   return GLMvector(static_cast<float>(atof(token1.c_str())),
      static_cast<float>(atof(token2.c_str())),
      static_cast<float>(atof(token3.c_str())));
}

GLMtexCoord ObjModel::convertToTexCoord(const string& token1, const string& token2)
{
   return GLMtexCoord(static_cast<float>(atof(token1.c_str())),
      static_cast<float>(atof(token2.c_str())));
}



unsigned int ObjModel::glmFindOrAddTexture(string file, bool displacementMap)
{
   // check whether a texture with this filename has already been loaded
   for(unsigned int i = 0; i < mTextures->size(); i++)
   {
      if(!file.compare(mTextures->at(i).fileName))
      {
         // found texture!
         // cout << "[ObjModel] glmFindOrAddTexture: found existing texture with filename " << file << endl;
         return i;
      }
   }

   // did not find texture	
   // load it
    cout << "[ObjModel] glmFindOrAddTexture: adding texture with filename " << file << endl;

   GLMtexture tex;
   int width, height;
   tex.fileName = file;
   tex.id = loadTextureFromImage(file, width, height, displacementMap);
   tex.width = width;
   tex.height = height;

   mTextures->push_back(tex);

   return (mTextures->size()-1);

}


unsigned int ObjModel::glmFindMaterial(const string& name)
{

   for(unsigned int i = 0; i<mMaterials->size();i++)
   {
      if(!name.compare(mMaterials->at(i).name))
         return i;
   }

   /* didn't find the name, so print a warning and return the default
   material (0). */	
   cerr << "glmFindMaterial():  can't find material " << name << endl;
   return 0;
}



GLuint ObjModel::loadTextureFromImage(string file, int &width, int &height, bool greyscale)
{
   // Use the SFML to load an image

   GLuint texture = 0;

   sf::Image image;
   if (!image.LoadFromFile(file))
      cerr << "Texture: FAILED loading " << file << endl;
   width = image.GetWidth();
   height = image.GetHeight();

   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
   if(greyscale)
   {
      sf::Uint8* greyData = new sf::Uint8[width * height];
      for(int y = 0; y < height; y++)
      {
         for(int x = 0; x < width; x++)
         {
            int pos = y * width + x;
            int mirrorPos = (height - 1 - y) * width + x;
            greyData[pos] = image.GetPixelsPtr()[4*mirrorPos];
           // cout << int(greyData[pos]) << endl;
         }
      }
      gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RED, width, height, GL_RED, GL_UNSIGNED_BYTE, greyData);
      delete[] greyData;
   }
   else
   {
      sf::Uint8* mirroredData = new sf::Uint8[4* width * height];
      for(int y = 0; y < height; y++)
      {
         for(int x = 0; x < width; x++)
         {
            int pos = y * width + x;
            int mirrorPos = (height - 1 - y) * width + x;
            mirroredData[4*pos]   = image.GetPixelsPtr()[4*mirrorPos];
            mirroredData[4*pos+1] = image.GetPixelsPtr()[4*mirrorPos+1];
            mirroredData[4*pos+2] = image.GetPixelsPtr()[4*mirrorPos+2];
            mirroredData[4*pos+3] = image.GetPixelsPtr()[4*mirrorPos+3];
           // cout << int(greyData[pos]) << endl;
         }
      }

      //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image.GetPixelsPtr());
      // GL_SRGB8_ALPHA8_EXT -- GL_RGBA8 assumes linear input texture
      gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA8, width, height, GL_RGBA, GL_UNSIGNED_BYTE, mirroredData);

      delete[] mirroredData;
   }

   return texture;

}


GLfloat	ObjModel::glmUnitize()
{
   GLfloat bb[6];
   glmBoundingBox(bb);

   GLfloat minX = bb[0];
   GLfloat minY = bb[1];
   GLfloat minZ = bb[2];
   GLfloat maxX = bb[3];
   GLfloat maxY = bb[4];
   GLfloat maxZ = bb[5];

   GLfloat cx, cy, cz, w, h, d;
   GLfloat scale;

   /* calculate model width, height, and depth */
   w = abs(maxX) + abs(minX);
   h = abs(maxY) + abs(minY);
   d = abs(maxZ) + abs(minZ);

   /* calculate center of the model */
   cx = (maxX + minX) / 2.0f;
   cy = (maxY + minY) / 2.0f;
   cz = (maxZ + minZ) / 2.0f;

   /* calculate unitizing scale factor */
   scale = 2.0f / max(max(w, h), d);

   /* translate around center then scale */
   for (unsigned int i = 1; i < mVertices->size(); i++)
   {
      mVertices->at(i).x -= cx;
      mVertices->at(i).y -= cy;
      mVertices->at(i).z -= cz;
      mVertices->at(i).x *= scale;
      mVertices->at(i).y *= scale;
      mVertices->at(i).z *= scale;
   }

   mUnitized = true;

   return scale;
}



GLvoid ObjModel::glmFacetNormals()
{
   assert(!mVertices->empty());

   GLMvector u;
   GLMvector v;

   /* clobber any old facetnormals */
   mFacetNorms->clear();

   // facet count starts with 1, so push 1 dummy back:
   GLMvector dummyFacetNormal;
   mFacetNorms->push_back(dummyFacetNormal);

   // visit all triangles 
   for (unsigned int i = 1; i < mTriangles->size(); i++) 
   {
      mTriangles->at(i).fnIndex = i;		

      u.x = mVertices->at(mTriangles->at(i).vIndices[1]).x
         - mVertices->at(mTriangles->at(i).vIndices[0]).x;

      u.y = mVertices->at(mTriangles->at(i).vIndices[1]).y
         - mVertices->at(mTriangles->at(i).vIndices[0]).y;

      u.z = mVertices->at(mTriangles->at(i).vIndices[1]).z
         - mVertices->at(mTriangles->at(i).vIndices[0]).z;

      v.x = mVertices->at(mTriangles->at(i).vIndices[2]).x
         - mVertices->at(mTriangles->at(i).vIndices[0]).x;

      v.y = mVertices->at(mTriangles->at(i).vIndices[2]).y
         - mVertices->at(mTriangles->at(i).vIndices[0]).y;

      v.z = mVertices->at(mTriangles->at(i).vIndices[2]).z
         - mVertices->at(mTriangles->at(i).vIndices[0]).z;

      glm::vec3 c = glm::cross(u,v);
      mFacetNorms->push_back(glm::normalize(c));    
   }

   assert(mTriangles->size() == mFacetNorms->size());
}


GLvoid ObjModel::glmVertexNormals(GLfloat angle, bool smoothingGroups)
{
   if(!mOwnVectorData && !mCopiedTriangles)
   {
      // for vertex normals this model has to hold its own
      // vertex normal indices
      vector<GLMtriangle>* refTriangles = this->mTriangles; // pointer to reference

      // copy data to a new instance
      mTriangles = new vector<GLMtriangle>;

      (*mTriangles) = (*refTriangles);
      mCopiedTriangles = true;

   }

   assert(mFacetNorms->size()==mTriangles->size());

   mComputedVertexNormals = true;
   mVertexNormalsAngle = angle;
   mVertexNormalsSmoothingGroups = smoothingGroups;


   /* calculate the cosine of the angle (in degrees) */
   GLfloat cos_angle = cos(angle * F_PI / 180.0f);

   /* nuke any previous normals */
   mNormals->clear();
   GLMvector dummyNormal;
   mNormals->push_back(dummyNormal);


   // user wants special treatment of smoothing groups
   // and there is more than the default "s off" group
   if(smoothingGroups && mSmoothGroups->size()>1)
   {

      /* a structure that will hold a list of triangle indices for each vertex */
      vector<unsigned int>* members = new vector<unsigned int>[mVertices->size()];

      // process all smoothinggroups

      for(unsigned int s = 0; s < mSmoothGroups->size(); s++)
      {
         // clear all members
         for(unsigned int m = 0; m < mVertices->size(); m++)
         {
            members[m].clear();
         }


         /* for every triangle in this smoothing group, push back its index  */
         for (unsigned int t = 0; t < mSmoothGroups->at(s).triangles.size(); t++)
         {
            // there are 3 vertices for this triangle
            // this triangle's index:
            int tIndex = mSmoothGroups->at(s).triangles.at(t);
            // vertex 0 index of this triangle: mTriangles->at(tIndex).vIndices[0]
            // add triangle index i to each triangle list
            members[mTriangles->at(tIndex).vIndices[0]].push_back(tIndex);
            members[mTriangles->at(tIndex).vIndices[1]].push_back(tIndex);
            members[mTriangles->at(tIndex).vIndices[2]].push_back(tIndex);
         }

         /* for each vertex in this smoothing group, calculate the average normal */

         int checkSum = 0;
         for(unsigned int v = 1; v < mVertices->size(); v++)
         {
            // vertices not in this smoothing group have empty triangle lists 
            if(members[v].empty())
               continue;

            // process this vertex
            vector<unsigned int> indices = members[v]; 

            if(indices.size()==1)
            {
               // vertex belongs only to 1 triangle

               // no smoothing necessary: use face normal as vertex normal
               mNormals->push_back(mFacetNorms->at(mTriangles->at(indices[0]).fnIndex));
               int lastIndex = mNormals->size() -1 ;

               if(mTriangles->at(indices[0]).vIndices[0] == v)
               {
                  mTriangles->at(indices[0]).vnIndices[0] = lastIndex;
               }
               else if(mTriangles->at(indices[0]).vIndices[1] == v)
               {
                  mTriangles->at(indices[0]).vnIndices[1] = lastIndex;
               }
               else if(mTriangles->at(indices[0]).vIndices[2] == v)
               {
                  mTriangles->at(indices[0]).vnIndices[2] = lastIndex;
               }
            }
            else
            {
               // smooth across all vertex' triangle face normals 
               glmComputeAndSetVertexNormal(indices, v, cos_angle);
            }

            checkSum++;
         }
      }
      delete[] members;
   }
   else
   {
      // process whole model without accessing smoothing groups

      /* a structure that will hold a list of triangle indices for each vertex */
      vector<unsigned int>* members = new vector<unsigned int>[mVertices->size()];
      // members[i] is a vector of triangle indices for vertex i

      /* for every triangle, push back its index  */
      for (unsigned int t = 1; t < mTriangles->size(); t++)
      {
         // there are 3 vertices for this triangle
         // this triangle: mTriangles->at(i)
         // vertex 0 index of this triangle: mTriangles->at(i).vIndices[0]
         // add triangle index i to each triangle list
         members[mTriangles->at(t).vIndices[0]].push_back(t);
         members[mTriangles->at(t).vIndices[1]].push_back(t);
         members[mTriangles->at(t).vIndices[2]].push_back(t);
      }


      /* for each vertex, calculate the average normal */

      for(unsigned int v = 1; v < mVertices->size(); v++)
      {
         // these are the collected triangle indices for this vertex v
         vector<unsigned int> indices = members[v]; 
         //if(indices.empty())
         //   cerr << "glmVertexNormals(): vertex w/o a triangle" << endl;

         if(indices.size()==1)
         {
            // vertex belongs only to 1 triangle

            // no smoothing necessary: use face normal as vertex normal
            mNormals->push_back(mFacetNorms->at(mTriangles->at(indices[0]).fnIndex));
            int lastIndex = mNormals->size() -1 ;

            if(mTriangles->at(indices[0]).vIndices[0] == v)
            {
               mTriangles->at(indices[0]).vnIndices[0] = lastIndex;
            }
            else if(mTriangles->at(indices[0]).vIndices[1] == v)
            {
               mTriangles->at(indices[0]).vnIndices[1] = lastIndex;
            }
            else if(mTriangles->at(indices[0]).vIndices[2] == v)
            {
               mTriangles->at(indices[0]).vnIndices[2] = lastIndex;
            }
         }
         else
         {
            // smooth across all vertex' triangle face normals 
            glmComputeAndSetVertexNormal(indices, v, cos_angle);
         }

      }

      delete[] members;

   }
   //cout << "[ObjModel glmVertexNormals] Have " << mNormals->size()-1 << " vertex normals." << endl;
   //cout << "[ObjModel glmVertexNormals] Have " << mVertices->size()-1 << " vertices." << endl;
}



GLvoid ObjModel::glmComputeAndSetVertexNormal(const vector<unsigned int>& indices, unsigned int v, GLfloat cos_angle)
{
   GLMvector averageNormal;
   GLuint avg=0;
   GLuint avgIndex=0;
   GLfloat dot;
   bool* averaged = new bool[indices.size()];

   // process triangles
   for(unsigned int i = 0; i < indices.size(); i++)
   {
      /* only average if the dot product of the angle between the two
      facet normals is greater than the cosine of the threshold
      angle -- or, said another way, the angle between the two
      facet normals is less than (or equal to) the threshold angle */
      dot = glm::dot(mFacetNorms->at(mTriangles->at(indices[i]).fnIndex),
         mFacetNorms->at(mTriangles->at(indices[0]).fnIndex));

      if (dot > cos_angle)
      {
         averaged[i] = true;
         // current triangle index:               indices[i]
         // current triangle:                     mTriangles->at(indices[i])
         // current triangle facet normal index : mTriangles->at(indices[i]).fnIndex 
         // current triangle facet normal :       mFacetNorms->at(mTriangles->at(indices[i]).fnIndex)
         averageNormal.x += mFacetNorms->at(mTriangles->at(indices[i]).fnIndex).x;
         averageNormal.y += mFacetNorms->at(mTriangles->at(indices[i]).fnIndex).y;
         averageNormal.z += mFacetNorms->at(mTriangles->at(indices[i]).fnIndex).z;
         avg = 1;            /* we averaged at least one normal! */
      } else
      {
         averaged[i]= false;
      }
   }
   // for each vertex:
   if (avg)
   {
      /* normalize the averaged normal */
      averageNormal = glm::normalize(averageNormal);

      /* add the normal to the vertex normals list */
      mNormals->push_back(averageNormal);
      avgIndex = mNormals->size()-1; // index of this normal
   }
   

   /* set the normal of this vertex in each triangle this vertex is in */
   // visit all triangles this vertex is in
   //   (triangle indices contained in vector indices)
   for(unsigned int i = 0; i<indices.size() ; i++)
   {
      //GLMtriangle currentTriangle = mTriangles->at(indices[i]);
      if (averaged[i])
      {
         /* if this node was averaged, use the average normal */

         // we have to check which of the 3 triangle vertices is currently processed
         // overwrite vertex normal index of currently processed vertex with average index

         if(mTriangles->at(indices[i]).vIndices[0] == v)
         {
            mTriangles->at(indices[i]).vnIndices[0] = avgIndex;
         }
         else if(mTriangles->at(indices[i]).vIndices[1] == v)
         {
            mTriangles->at(indices[i]).vnIndices[1] = avgIndex;
         }
         else if(mTriangles->at(indices[i]).vIndices[2] == v)
         {
            mTriangles->at(indices[i]).vnIndices[2] = avgIndex;
         }

      } else
      {

         /* if this node wasn't averaged, use the facet normal */

         mNormals->push_back(mFacetNorms->at(mTriangles->at(indices[i]).fnIndex));
         int lastIndex = mNormals->size() -1 ;

         if(mTriangles->at(indices[i]).vIndices[0] == v)
         {
            mTriangles->at(indices[i]).vnIndices[0] = lastIndex;
         }
         else if(mTriangles->at(indices[i]).vIndices[1] == v)
         {
            mTriangles->at(indices[i]).vnIndices[1] = lastIndex;
         }
         else if(mTriangles->at(indices[i]).vIndices[2] == v)
         {
            mTriangles->at(indices[i]).vnIndices[2] = lastIndex;
         }

      }
   }
   delete[] averaged;

}




GLboolean ObjModel::glmEqual(const GLMvector& u, const GLMvector& v, GLfloat epsilon)
{
   if (abs(u.x - v.x) < epsilon &&
      abs(u.y - v.y) < epsilon &&
      abs(u.z - v.z) < epsilon) 
   {
      return GL_TRUE;
   }
   return GL_FALSE;
}


GLvoid ObjModel::glmCenter()
{
   GLfloat bb[6];
   glmBoundingBox(bb);

   GLfloat minX = bb[0];
   GLfloat minY = bb[1];
   GLfloat minZ = bb[2];
   GLfloat maxX = bb[3];
   GLfloat maxY = bb[4];
   GLfloat maxZ = bb[5];

   /* calculate model width, height, and depth */
   float w = abs(maxX) + abs(minX);
   float h = abs(maxY) + abs(minY);
   float d = abs(maxZ) + abs(minZ);

   /* calculate center of the model */
   float cx = (maxX + minX) / 2.0f;
   float cy = (maxY + minY) / 2.0f;
   float cz = (maxZ + minZ) / 2.0f;

   /* translate around center then scale */
   for (unsigned int i = 1; i < mVertices->size(); i++)
   {
      mVertices->at(i).x -= cx;
      mVertices->at(i).y -= cy;
      mVertices->at(i).z -= cz;
   }

   mCentered = true;
}

GLvoid ObjModel::glmBoundingBox(GLfloat* boundingBox)
{

   GLfloat maxX, minX, maxY, minY, maxZ, minZ;

   assert(!mVertices->empty());
   assert(boundingBox);

   /* get the max/mins from first vertex*/
   maxX = minX = mVertices->at(1).x;
   maxY = minY = mVertices->at(1).y;
   maxZ = minZ = mVertices->at(1).z;
   for (unsigned int i = 1; i < mVertices->size(); i++) {
      if (maxX < mVertices->at(i).x)
         maxX = mVertices->at(i).x;
      if (minX > mVertices->at(i).x)
         minX = mVertices->at(i).x;

      if (maxY < mVertices->at(i).y)
         maxY = mVertices->at(i).y;
      if (minY > mVertices->at(i).y)
         minY = mVertices->at(i).y;

      if (maxZ < mVertices->at(i).z)
         maxZ = mVertices->at(i).z;
      if (minZ > mVertices->at(i).z)
         minZ = mVertices->at(i).z;
   }

   boundingBox[0] = minX;
   boundingBox[1] = minY;
   boundingBox[2] = minZ;
   boundingBox[3] = maxX;
   boundingBox[4] = maxY;
   boundingBox[5] = maxZ;

}




GLvoid ObjModel::glmDrawImmediate(GLuint mode)
{
   assert(!mVertices->empty());


   /* do a bit of warning */
   if (mode & GLM_FLAT && mFacetNorms->size()<=1)
   {
      cerr << "glmDraw() warning: flat render mode requested "
         << "with no facet normals defined."<<endl;
      mode &= ~GLM_FLAT;
   }
   if (mode & GLM_SMOOTH && mNormals->size()<=1)
   {
      cerr << "glmDraw() warning: smooth render mode requested "
         << "with no normals defined." << endl;
      mode &= ~GLM_SMOOTH;
   }
   if (mode & GLM_TEXTURE_ATLAS 
      && ((mAtlasTexCoords == 0) || ((mAtlasTexCoords != 0) && mAtlasTexCoords->size()<=1)) )
   {
      cerr << "glmDraw() warning: atlas texture render mode requested "
         << "with no atlas texture coordinates defined." << endl;
      mode &= ~GLM_TEXTURE_ATLAS;
   }
   if (mode & GLM_FLAT && mode & GLM_SMOOTH)
   {
      cerr << "glmDraw() warning: flat render mode requested "
         << " and smooth render mode requested (using smooth)." << endl;
      mode &= ~GLM_FLAT;
   }
   if (mode & GLM_COLOR && mMaterials->size()==1)
   {
      cerr << "glmDraw() warning: color render mode requested "
         << " with no materials defined. Use default material.\n" << endl;
      //mode &= ~GLM_COLOR;
   }
   if (mode & GLM_MATERIAL && mMaterials->size()==1)
   {
      cerr << "glmDraw() warning: material render mode requested "
         << " with no materials defined. Use default material." << endl;
   }
   if (mode & GLM_COLOR && mode & GLM_MATERIAL)
   {
      cerr << "glmDraw() warning: color and material render mode requested "
         << "using only color mode. " << endl;
      mode &= ~GLM_MATERIAL;
   }
   if (mode & GLM_COLOR)
   {
       glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
       glEnable(GL_COLOR_MATERIAL);
   }
   else if (mode & GLM_MATERIAL)
       glDisable(GL_COLOR_MATERIAL);

   // only for standard pipeline necessary
   //if (mode & GLM_TEXTURE)
   //{
   //   glEnable(GL_TEXTURE_2D);
   //   glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
   //}

   // iterate through groups
   for(unsigned int g = 0; g < mGroups->size(); g++)
   {
      GLMgroup currentGroup = mGroups->at(g);
      GLMmaterial currentMaterial = mMaterials->at(0); // default material

      // if materials defined           
      if(mMaterials->size() > 1) 
      {
         currentMaterial = mMaterials->at(currentGroup.materialIndex);
      }
      //std::cout << currentMaterial.name << std::endl;
      //std::cout << currentMaterial.diffuse[0] << " " << currentMaterial.diffuse[1] << " " << currentMaterial.diffuse[2] << std::endl;

      //		cout << endl << "Current group " << currentGroup.name << endl;
      //		cout << "Accessing mMaterials (size: " << mMaterials->size() << ") at: " << currentGroup.materialIndex << endl;
      //		cout << "Current material " << currentMaterial.name << endl;

      if (mode & GLM_MATERIAL) 
      {  
         glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, currentMaterial.diffuse);
         //glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, currentMaterial.ambient);
         //glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, currentMaterial.specular);
         //glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, currentMaterial.shininess);
      }

      if(mode & GLM_TEXTURE || mode & GLM_TEXTURE_FIX)
      {
         glActiveTexture(GL_TEXTURE0);
      }
      if (mode & GLM_TEXTURE) 
      {
         if(currentMaterial.has_map_Kd)
         {
            glBindTexture(GL_TEXTURE_2D, mTextures->at(currentMaterial.index_map_Kd).id);	
         }
         else
         {
            glBindTexture(GL_TEXTURE_2D, TexturePool::getTexture("whitePixel"));	
         }
      }

      else if ( mode & GLM_TEXTURE_FIX )
      {
         glBindTexture(GL_TEXTURE_2D, TexturePool::getTexture("whitePixel"));
      }

      if (mode & GLM_COLOR)
      {
         glColor4fv(currentMaterial.diffuse);
      }

      glBegin(GL_TRIANGLES);

      if(mode & GLM_TEXTURE_FIX)
      {
         glMultiTexCoord2f(GL_TEXTURE0, 0, 0);

      }

      for (unsigned int t = 0; t < currentGroup.triangles.size(); t++)
      {
          GLMtriangle triangle = mTriangles->at(currentGroup.triangles.at(t));

         if (mode & GLM_FLAT)
         {
            glNormal3f(mFacetNorms->at(triangle.fnIndex).x,
               mFacetNorms->at(triangle.fnIndex).y,
               mFacetNorms->at(triangle.fnIndex).z);
         }

         for (int idx = 0; idx <= 2; idx++)
         {
            if (mode & GLM_SMOOTH)
            {
               glNormal3f(mNormals->at(triangle.vnIndices[idx]).x,
                  mNormals->at(triangle.vnIndices[idx]).y,
                  mNormals->at(triangle.vnIndices[idx]).z);
            }
            if (mode & GLM_TEXTURE)
            {
                glMultiTexCoord2f(GL_TEXTURE0,
                  mTexCoords->at(triangle.vtIndices[idx]).s,
                  mTexCoords->at(triangle.vtIndices[idx]).t);
            }
            if (mode & GLM_TEXTURE_ATLAS)
            {
               glMultiTexCoord2f(GL_TEXTURE0,
                  mAtlasTexCoords->at(triangle.vatIndices[idx]).s,
                  mAtlasTexCoords->at(triangle.vatIndices[idx]).t);
            }

             glVertex3f(mVertices->at(triangle.vIndices[idx]).x,
               mVertices->at(triangle.vIndices[idx]).y,
               mVertices->at(triangle.vIndices[idx]).z);
         }


      }
      glEnd();

   }

   // reset OpenGL states
   if (mode & GLM_TEXTURE)
   {
      glDisable(GL_TEXTURE_2D);
   }
}


GLvoid ObjModel::draw(GLuint mode)
{
   if( !(mode & GLM_TEXTURE_ATLAS) && ((mode & GLM_COLOR) || (mode & GLM_MATERIAL)) && !(mode & GLM_TEXTURE))
   {
      mode |= GLM_TEXTURE_FIX; 
   }

   glmDrawList(mode);
}

GLvoid ObjModel::glmDrawList(GLuint mode)
{
   // search for list corresponding to mode
   map<GLuint, GLuint>::iterator displayListIterator = mDisplayLists.find(mode);

   // not found! => insert
   if(displayListIterator == mDisplayLists.end())
   {
      cout << "[ObjModel] DL: " << mPathModel << endl;
      //cout << "           inserting displayList for mode " << drawModeToString(mode) << endl;
      mDisplayLists[mode] = glmList(mode);		
      displayListIterator = mDisplayLists.find(mode);
   }
   // else found

   glCallList(displayListIterator->second);

}

string ObjModel::drawModeToString(GLuint mode)
{
   if(mode == GLM_NONE)
   {
      return "GLM_NONE";
   }
   else
   {
      string s;
      if(mode & GLM_FLAT)
      {
         s += "GLM_FLAT ";
      }
      if(mode & GLM_SMOOTH)
      {
         s += "GLM_SMOOTH ";
      }
      if(mode & GLM_TEXTURE)
      {
         s += "GLM_TEXTURE ";
      }
      if(mode & GLM_COLOR)
      {
         s += "GLM_COLOR ";
      }
      if(mode & GLM_MATERIAL)
      {
         s += "GLM_MATERIAL ";
      }
      if(mode & GLM_TEXTURE_FIX)
      {
         s += "GLM_TEXTURE_FIX ";
      }
      if(mode & GLM_TEXTURE_ATLAS)
      {
         s += "GLM_TEXTURE_ATLAS ";
      }
      return s;

   }
   
}

GLuint ObjModel::glmList(GLuint mode)
{
   GLuint displayList = glGenLists(1);
   glNewList(displayList, GL_COMPILE);
   glmDrawImmediate(mode);
   glEndList();
   return displayList;
}



GLvoid ObjModel::glmScale(GLfloat scale)
{
   for (unsigned int i = 1; i < mVertices->size(); i++)
   {
      mVertices->at(i).x *= scale;
      mVertices->at(i).y *= scale;
      mVertices->at(i).z *= scale;
   }

   mScaleFactor = scale;
}

void ObjModel::createWhitePixelTexture()
{

   GLfloat data[] = {1, 1, 1};
   GLuint whitePixelTex;
	glGenTextures(1, &whitePixelTex);
	glBindTexture(GL_TEXTURE_2D, whitePixelTex);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_FLOAT, data);

   TexturePool::addTexture("whitePixel", whitePixelTex);

}
