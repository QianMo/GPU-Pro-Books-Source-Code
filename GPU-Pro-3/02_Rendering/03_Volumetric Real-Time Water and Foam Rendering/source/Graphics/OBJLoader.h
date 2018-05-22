#ifndef __OBJLOADER__H__
#define __OBJLOADER__H__


class OBJLoader
{
public:

	OBJLoader(void);
	~OBJLoader(void);

	void Exit(void);

	// load a wavefront obj returns number of triangles that were loaded.  Data is persists until the class is destructed.
	unsigned int LoadObj(const char *fname, bool textured);

	int          mVertexCount;
	int          mTriCount;
	int          *mIndices;
	float        *mVertices;
	float		 *mNormals;
	float        *mTexCoords;
};

#endif
