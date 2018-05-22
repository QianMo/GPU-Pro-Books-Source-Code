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

#ifndef OBJMODELSTRUCTS
#define OBJMODELSTRUCTS

#include "glm/glm.hpp" // OpenGL Mathematics Library

/// GLMvector: defines a vertex or normal (x,y,z).
typedef glm::vec3 GLMvector;


/// GLMtexCoord: defines a texture coordinate (s,t).
typedef glm::vec2 GLMtexCoord;


/// GLMtexture: Structure that defines a texture.
struct GLMtexture
{
  string fileName;
  GLuint id;                   
  int width;		
  int height;
};


/// GLMmaterial: Structure that defines a material in a model. 
struct GLMmaterial
{
  string name;                  /* name of material */
  int id;
  GLfloat diffuse[4];           /* diffuse component */
  GLfloat ambient[4];           /* ambient component */
  GLfloat specular[4];          /* specular component */
  GLfloat emmissive[4];         /* emmissive component */
  GLfloat shininess;            /* specular exponent */
  GLboolean has_map_Kd;
  unsigned int index_map_Kd;    ///< index pointing to diffuse color map
  GLfloat senderScaleFactor; ///< Scales the amount of indirect light an object is reflecting
};

/// GLMtriangle: Structure that defines a triangle in a model.
struct GLMtriangle
{
  unsigned int vIndices[3];     /* triangle vertex indices */
  unsigned int vnIndices[3];    /* triangle normal indices */
  unsigned int vtIndices[3];    /* triangle texcoord indices*/
  unsigned int vatIndices[3];    /* triangle atlas texcoord indices*/
  unsigned int fnIndex;         /* index of triangle facet normal */
  bool smooth;					/* true = belongs to smoothing group > 0 */
};


/// GLMgroup: Structure that defines a group in a model.
struct GLMgroup
{
  string                name;           /* name of this group */
  vector<unsigned int>  triangles;      /* array of triangle indices for vector triangles*/
  unsigned int          materialIndex;  /* index to materials */
};


/// GLMsmoothGroup: Structure that defines a smoothing group
struct GLMsmoothGroup
{
  unsigned int          id;             /* id of this smoothing group */
  vector<unsigned int>  triangles;      /* array of triangle indices for vector triangles*/
};


/// Vertex Buffer Object Data
struct VBOData
{
   float vx, vy, vz; // 12 byte - vertices
   float nx, ny, nz; // 12 byte - normals
 //  float r, g, b;    // 12 byte - color
   float s_color, t_color;     // 8 byte  - color texture coordinates
   float s_atlas, t_atlas;     // 8 byte  - atlas texture coordinates
 //  float padding[3]; // 12 byte - padding for 64 byte structure
};

/// Helping structure for building the VBO Data
struct VertexIndices
{
   unsigned int normalIndex;
   unsigned int texCoordIndex;
   unsigned int atlasTexCoordIndex;
   unsigned int renderIndex;
};


#endif
