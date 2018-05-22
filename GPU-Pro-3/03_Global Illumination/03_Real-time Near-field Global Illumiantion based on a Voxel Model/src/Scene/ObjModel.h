/*    

     BASED ON:

      glm.h
      Nate Robins, 1997, 2000
      nate@pobox.com, http://www.pobox.com/~nate
 
      Wavefront OBJ model file format reader/writer/manipulator.

      Includes routines for generating smooth normals with
      preservation of edges, welding redundant vertices & texture
      coordinate generation (spheremap and planar projections) + more.

 */


#ifndef OBJMODEL_H
#define OBJMODEL_H

#define GLM_NONE          (0)            ///< render with only vertices
#define GLM_FLAT          (1 << 0)       ///< render with facet normals
#define GLM_SMOOTH        (1 << 1)       ///< render with vertex normals
#define GLM_TEXTURE       (1 << 2)       ///< render with texture coords
#define GLM_MATERIAL      (1 << 3)       ///< render with materials
#define GLM_COLOR         (1 << 4)       ///< render with colors 
#define GLM_TEXTURE_ATLAS (1 << 5)       ///< render with atlas texture coords 
#define GLM_TEXTURE_FIX   (1 << 7)       ///< handles non-textured materials (diffuseTex in gbuffer shader)

// STL
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <cassert>
#include <list>

using namespace std;

// SFML
#include "SFML/Graphics.hpp" // for image loading

#include "OpenGL.h"
#include "Scene/ObjModelStructs.h"

   
class ShaderProgram;

class ObjModel
{
public:

   /// Reads a model description from a Wavefront OBJ file.
   /// \param file Path to obj-model file
   /// \param usemtlGroups Indicates whether to group materials by usemtl-lines
   /// or to group by g-lines (uses last called usemtl as material for active group g).
   ///
   ObjModel::ObjModel(string file, bool usemtlGroups = true);


   /// Reads the model vertices from a Wavefront OBJ file
   /// and takes faces, indices etc. from the refModel.
   ObjModel::ObjModel(string file, const ObjModel* refModel);

   /// Destructor.
   ~ObjModel();

   /// Extracts texture coordinates from a Wavefront OBJ file.
   /// These are used as texture atlas coordinates for this model.
   /// Requires same vertex and face order as original model.
   /// \param file Path to obj-model file
   ///
   GLvoid addAtlasTextureCoordinates(string file);

   /// Assigns ObjModel m's texture atlas coordinates vector pointer 
   /// to this model's vector pointer.
   // GLvoid useAtlasTextureCoordinatesFrom(const ObjModel* m);


   /// Renders the model to the current OpenGL context using the
   /// mode specified.
   ///
   /// \param mode  - a bitwise OR of values describing what is to be rendered.
   ///             GLM_NONE     -  render with only vertices
   ///             GLM_FLAT     -  render with facet normals
   ///             GLM_SMOOTH   -  render with vertex normals
   ///             GLM_TEXTURE  -  render with texture coords
   ///             GLM_COLOR    -  render with colors (color material)
   ///             GLM_MATERIAL -  render with materials
   ///             GLM_COLOR and GLM_MATERIAL should not both be specified.  
   ///             GLM_FLAT and GLM_SMOOTH should not both be specified.  
   ///
   GLvoid glmDrawImmediate(GLuint mode);

   /// Generates and uses a display list for drawing the model using
   /// the mode specified.
   /// 
   /// \param mode - a bitwise OR of values describing what is to be rendered.
   ///            GLM_NONE    -  render with only vertices
   ///            GLM_FLAT    -  render with facet normals
   ///            GLM_SMOOTH  -  render with vertex normals
   ///            GLM_TEXTURE -  render with texture coords
   ///            GLM_FLAT and GLM_SMOOTH should not both be specified.  
   GLvoid glmDrawList(GLuint mode);


   /// Generates a display list for drawing the model using the mode specified.
   GLuint glmList(GLuint mode);

   GLvoid draw(GLuint mode);

   /// Reads in an image file.
   /// \param[in] file The texture image file name.
   /// \param[out] width The width of the texture
   /// \param[out] height The height of the texture
   /// Returns an OpenGL Texture Handle.
   ///
   GLuint loadTextureFromImage(string file, int& width, int& height, bool greyscale);

   /// Generates facet normals for a model (by taking the
   /// cross product of the two vectors derived from the sides of each
   /// triangle).  Assumes a counter-clockwise winding.
   /// Also records all triangle normals weighted by area 
   /// in an orientation histogram (8 bit (theta, phi) 
   ///
   GLvoid glmFacetNormals();


   /// Generates smooth vertex normals for a model.
   /// First builds a list of all the triangles each vertex is in.  Then
   /// loops through each vertex in the the list averaging all the facet
   /// normals of the triangles each vertex is in.  Finally, sets the
   /// normal index in the triangle for the vertex to the generated smooth
   /// normal.  If the dot product of a facet normal and the facet normal
   /// associated with the first triangle in the list of triangles the
   /// current vertex is in is greater than the cosine of the angle
   /// parameter to the function, that facet normal is not added into the
   /// average normal calculation and the corresponding vertex is given
   /// the facet normal.  This tends to preserve hard edges.  The angle to
   /// use depends on the model, but 90 degrees is usually a good start.
   /// \param angle - maximum angle (in degrees) to smooth across
   /// \param smoothingGroups true if only the smoothing groups of the model should be smoothed,
   ///        false if the whole model will be smoothed, neglecting all smoothing groups
   ///
   GLvoid glmVertexNormals(GLfloat angle, bool smoothingGroups);

   /// "unitize" a model by translating it to the origin and
   /// scaling it to fit in a unit cube around the origin.  Returns the
   /// scale factor used.
   ///
   GLfloat   glmUnitize();

   /// Calculates the axis-aligned boundingBox 
   /// (minX, minY, minZ, maxX, maxY, maxZ) 
   /// of the model.
   /// \param[out] boundingBox - array of 6 GLfloats (GLfloat boundingBox[6])
   GLvoid glmBoundingBox(GLfloat* boundingBox);

   /// Centers the model around the origin.
   ///
   GLvoid glmCenter();


   /// Scales a model by a given amount.
   /// \param scale Scalefactor (0.5 = half as large, 2.0 = twice as large)
   ///
   GLvoid glmScale(GLfloat scale);

   int getTriangleCount() const { return mTriangles->size() - 1; }
   const vector<GLMtriangle>* getTriangles() const { return mTriangles; }
   const vector<GLMvector>* getVertices() const { return mVertices; }


   bool hasComputedVertexNormals() const { return mComputedVertexNormals; }
   float getVertexNormalsAngle() const { return mVertexNormalsAngle; }
   bool usesVertexNormalsSmoothingGroups() const { return mVertexNormalsSmoothingGroups; }
   bool isUnitized() const { return mUnitized; }
   bool isCentered() const { return mCentered; }
   float getScaleFactor() const { return mScaleFactor; } // returns the scaling factor of last call to glmScale

   string getPathModel() const { return mPathModel; }
   string getPathAtlas() const { return mPathAtlas; }

   static string drawModeToString(GLuint mode);

   // workaround for non-textured triangles
   static void createWhitePixelTexture();

private:

   ObjModel();

   /// Initializes some member variables with default values.
   void initSettings();

   /// Reads a model description from a Wavefront OBJ file.
   ///
   /// \param filename - name of the file containing the Wavefront .OBJ format data.  
   GLvoid glmReadOBJ(string file);

   /// Reads model vertices from a Wavefront OBJ file.
   ///
   /// \param filename - name of the file containing the Wavefront .OBJ format data.  
   GLvoid glmReadOBJVertices(string file);

   /// Reads a wavefront material library file.
   ///
   /// \param file  - name of the material library
   ///
   GLvoid glmReadMTL(string file);

   /// Returns the directory given a path.
   ///
   /// path - filesystem path
   ///
   string glmDirName(string path);


   /// Returns the index of a certain group called name.
   /// If a group with this name already exists in vector groups,
   /// return its index.
   /// If such a group does not exist, push it back and 
   /// return the index of this new group.
   /// \param name Group name to be found or added as new empty group
   ///
   unsigned int glmFindOrAddGroup(const string& name);

   /// Returns vector index of the smooth group identified by id.
   /// If a group with this id already exists in vector smoothGroups,
   /// return its index.
   /// If such a group does not exist, push it back and 
   /// return the index of this new group.
   /// \param id Smoothing group id
   ///
   unsigned int glmFindOrAddSmoothGroup(unsigned int id);

   /// Returns whether groups contains a group called name.
   /// If found: write group index to index.
   /// \param[in] name Group name
   /// \param[out] index Group index of found group
   ///
   bool glmFindGroup(string name, unsigned int& index);

   /// Returns whether smoothGroups contains a smooth group with given id.
   /// If found: write group index to index
   /// \param[in] name Smoothing Group id
   /// \param[out] index Smoothing group index of found smoothing group
   ///
   bool glmFindSmoothGroup(unsigned int id, unsigned int& index);

   
   /// Search for a texture in model's vector<GLMtexture> with fileName file.
   /// If found: return this texture. 
   /// If not found: add a new texture, load it.
   /// Returns this texture's index (entry in vector textures).
   /// \param file The texture image file name
   unsigned int glmFindOrAddTexture(string file, bool displacementMap);

   /// Deletes all textures that were loaded for this model.
   ///
   GLvoid deleteTextures();

   /// Deletes all display lists that were created for this model.
   ///
   GLvoid deleteDisplayLists();

   /// Finds and returns the index of a named material in the model
   ///
   unsigned int glmFindMaterial(const string& name);

   /// Parsing methods
   GLMvector convertToVector(const string& token1, const string& token2, const string& token3);
   GLMtexCoord convertToTexCoord(const string& token1, const string& token2);

   /// Computes a smoothed vertex normal for a given set of trianges, adds it to the model structure
   /// and sets the corresponding normal indices.
   /// \param[in] indices A set of triangle indices (entry number in vector triangles)
   /// \param[in] v The index of the current process vertex
   /// \param[in] cos_angle Cosinus of the smoothing angle
   GLvoid glmComputeAndSetVertexNormal(const vector<unsigned int>& indices, unsigned int v, GLfloat cos_angle);

   /// Compares two vectors and returns GL_TRUE if they are
   /// equal (within a certain threshold) or GL_FALSE if not. 
   /// \param u - struct GLMvector (x,y,z)
   /// \param v - struct GLMvector (x,y,z)
   /// \param epsilon Threshold for comparison, an epsilon
   ///        that works fairly well is 0.000001.
   ///
   GLboolean glmEqual(const GLMvector& u, const GLMvector& v, GLfloat epsilon);


   // Member data
   string   mPathModel;            ///< path to this model's obj file
   string   mPathAtlas;            ///< path to this model's atlas (obj file)
   string   mMtlLibName;          ///< name of the material library 

   vector<GLMvector>* mVertices;     ///< array of vertices
   vector<GLMvector>* mNormals;      ///< array of normals
   vector<GLMtexCoord>* mTexCoords;  ///< array of texture coordinates
   vector<GLMtexCoord>* mAtlasTexCoords;  ///< array of atlas texture coordinates
   vector<GLMvector>*   mFacetNorms;   ///< array of facetnorms
   vector<GLMtriangle>* mTriangles;  ///< array of triangles
   vector<GLMmaterial>* mMaterials;  ///< array of materials

   vector<GLMgroup>*       mGroups;       ///< material groups
   vector<GLMsmoothGroup>* mSmoothGroups; ///< smoothing groups
   vector<GLMtexture>*     mTextures;     ///< loaded textures

   GLfloat mPosition[3];          ///< position of the model 

   map<GLuint, GLuint> mDisplayLists;  ///< mode, display list

   vector<unsigned int*> mGroupsIndices;

   bool mUsemtlGroups;

   bool mOwnVectorData; ///< Does this model have own data for texCoords, atlasTexCoords, triangles, materials, groups,
                        /// smoothGroups, textures

   bool mCopiedTriangles; ///< Is vector pointer mTriangles pointing to a reference model 
                          /// or is triangle data copied to a new instance?

   // Settings

   bool mComputedVertexNormals; ///< Vertex normals have been computed
   float mVertexNormalsAngle; ///< Threshold for vertex normal computation
   bool mVertexNormalsSmoothingGroups; ///< Use smoothing groups
   bool mUnitized; ///< Does this model fit into a centered unit cube
   bool mCentered; ///< Is this model centered around (0, 0, 0)
   float mScaleFactor; ///< Scaling factor for all dimensions

};


#endif
