#include <stdio.h>
//#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>

#include "OBJLoader.h"

#include <vector>
#include <math.h>

typedef std::vector< int > IntVector;
typedef std::vector< float > FloatVector;

/*******************************************************************/
/******************** InParser.h  ********************************/
/*******************************************************************/
class InPlaceParserInterface
{
public:
	virtual int ParseLine(int lineno,int argc,const char **argv) =0;  // return TRUE to continue parsing, return FALSE to abort parsing process
};

enum SeparatorType
{
	ST_DATA,        // is data
	ST_HARD,        // is a hard separator
	ST_SOFT,        // is a soft separator
	ST_EOS          // is a comment symbol, and everything past this character should be ignored
};

class InPlaceParser
{
public:
	InPlaceParser(void)
	{
		Init();
	}

	InPlaceParser(char *data,int len)
	{
		Init();
		SetSourceData(data,len);
	}

	InPlaceParser(const char *fname)
	{
		Init();
		SetFile(fname);
	}

	~InPlaceParser(void);

	void Init(void)
	{
		mQuoteChar = 34;
		mData = 0;
		mLen  = 0;
		mMyAlloc = false;
		for (int i=0; i<256; i++)
		{
			mHard[i] = ST_DATA;
			mHardString[i*2] = (char)i;
			mHardString[i*2+1] = 0;
		}
		mHard[0]  = ST_EOS;
		mHard[32] = ST_SOFT;
		mHard[9]  = ST_SOFT;
		mHard[13] = ST_SOFT;
		mHard[10] = ST_SOFT;
	}

	void SetFile(const char *fname); // use this file as source data to parse.

	void SetSourceData(char *data,int len)
	{
		mData = data;
		mLen  = len;
		mMyAlloc = false;
	};

	int  Parse(InPlaceParserInterface *callback); // returns true if entire file was parsed, false if it aborted for some reason

	int ProcessLine(int lineno,char *line,InPlaceParserInterface *callback);

	const char ** GetArglist(char *source,int &count); // convert source string into an arg list, this is a destructive parse.

	void SetHardSeparator(char c) // add a hard separator
	{
		mHard[c] = ST_HARD;
	}

	void SetHard(char c) // add a hard separator
	{
		mHard[c] = ST_HARD;
	}


	void SetCommentSymbol(char c) // comment character, treated as 'end of string'
	{
		mHard[c] = ST_EOS;
	}

	void ClearHardSeparator(char c)
	{
		mHard[c] = ST_DATA;
	}


	void DefaultSymbols(void); // set up default symbols for hard seperator and comment symbol of the '#' character.

	bool EOS(char c)
	{
		if ( mHard[c] == ST_EOS )
		{
			return true;
		}
		return false;
	}

	void SetQuoteChar(char c)
	{
		mQuoteChar = c;
	}

private:


	inline char * AddHard(int &argc,const char **argv,char *foo);
	inline bool   IsHard(char c);
	inline char * SkipSpaces(char *foo);
	inline bool   IsWhiteSpace(char c);
	inline bool   IsNonSeparator(char c); // non seperator,neither hard nor soft

	bool   mMyAlloc; // whether or not *I* allocated the buffer and am responsible for deleting it.
	char  *mData;  // ascii data to parse.
	int    mLen;   // length of data
	SeparatorType  mHard[256];
	char   mHardString[256*2];
	char           mQuoteChar;
};

/*******************************************************************/
/******************** InParser.cpp  ********************************/
/*******************************************************************/
void InPlaceParser::SetFile(const char *fname)
{
	if ( mMyAlloc )
	{
		free(mData);
	}
	mData = 0;
	mLen  = 0;
	mMyAlloc = false;

	FILE *fph;
	fopen_s(&fph, fname, "rb");

	if ( fph )
	{
		fseek(fph,0L,SEEK_END);
		mLen = ftell(fph);
		fseek(fph,0L,SEEK_SET);
		if ( mLen )
		{
			mData = (char *) malloc(sizeof(char)*(mLen+1));
			int ok = (int)fread(mData, mLen, 1, fph);
			if ( !ok )
			{
				free(mData);
				mData = 0;
			}
			else
			{
				mData[mLen] = 0; // zero byte terminate end of file marker.
				mMyAlloc = true;
			}
		}
		fclose(fph);
	}
}

InPlaceParser::~InPlaceParser(void)
{
	if ( mMyAlloc )
	{
		free(mData);
	}
}

#define MAXARGS 512

bool InPlaceParser::IsHard(char c)
{
	return mHard[c] == ST_HARD;
}

char * InPlaceParser::AddHard(int &argc,const char **argv,char *foo)
{
	while ( IsHard(*foo) )
	{
		const char *hard = &mHardString[*foo*2];
		if ( argc < MAXARGS )
		{
			argv[argc++] = hard;
		}
		foo++;
	}
	return foo;
}

bool   InPlaceParser::IsWhiteSpace(char c)
{
	return mHard[c] == ST_SOFT;
}

char * InPlaceParser::SkipSpaces(char *foo)
{
	while ( !EOS(*foo) && IsWhiteSpace(*foo) ) foo++;
	return foo;
}

bool InPlaceParser::IsNonSeparator(char c)
{
	if ( !IsHard(c) && !IsWhiteSpace(c) && c != 0 ) return true;
	return false;
}


int InPlaceParser::ProcessLine(int lineno,char *line,InPlaceParserInterface *callback)
{
	int ret = 0;

	const char *argv[MAXARGS];
	int argc = 0;

	char *foo = line;

	while ( !EOS(*foo) && argc < MAXARGS )
	{

		foo = SkipSpaces(foo); // skip any leading spaces

		if ( EOS(*foo) ) break;

		if ( *foo == mQuoteChar ) // if it is an open quote
		{
			foo++;
			if ( argc < MAXARGS )
			{
				argv[argc++] = foo;
			}
			while ( !EOS(*foo) && *foo != mQuoteChar ) foo++;
			if ( !EOS(*foo) )
			{
				*foo = 0; // replace close quote with zero byte EOS
				foo++;
			}
		}
		else
		{

			foo = AddHard(argc,argv,foo); // add any hard separators, skip any spaces

			if ( IsNonSeparator(*foo) )  // add non-hard argument.
			{
				bool quote  = false;
				if ( *foo == mQuoteChar )
				{
					foo++;
					quote = true;
				}

				if ( argc < MAXARGS )
				{
					argv[argc++] = foo;
				}

				if ( quote )
				{
					while (*foo && *foo != mQuoteChar ) foo++;
					if ( *foo ) *foo = 32;
				}

				// continue..until we hit an eos ..
				while ( !EOS(*foo) ) // until we hit EOS
				{
					if ( IsWhiteSpace(*foo) ) // if we hit a space, stomp a zero byte, and exit
					{
						*foo = 0;
						foo++;
						break;
					}
					else if ( IsHard(*foo) ) // if we hit a hard separator, stomp a zero byte and store the hard separator argument
					{
						const char *hard = &mHardString[*foo*2];
						*foo = 0;
						if ( argc < MAXARGS )
						{
							argv[argc++] = hard;
						}
						foo++;
						break;
					}
					foo++;
				} // end of while loop...
			}
		}
	}

	if ( argc )
	{
		ret = callback->ParseLine(lineno, argc, argv );
	}

	return ret;
}

int  InPlaceParser::Parse(InPlaceParserInterface *callback) // returns true if entire file was parsed, false if it aborted for some reason
{
	assert( callback );
	if ( !mData ) return 0;

	int ret = 0;

	int lineno = 0;

	char *foo   = mData;
	char *begin = foo;


	while ( *foo )
	{
		if ( *foo == 10 || *foo == 13 )
		{
			lineno++;
			*foo = 0;

			if ( *begin ) // if there is any data to parse at all...
			{
				int v = ProcessLine(lineno,begin,callback);
				if ( v ) ret = v;
			}

			foo++;
			if ( *foo == 10 ) foo++; // skip line feed, if it is in the carraige-return line-feed format...
			begin = foo;
		}
		else
		{
			foo++;
		}
	}

	lineno++; // lasst line.

	int v = ProcessLine(lineno,begin,callback);
	if ( v ) ret = v;
	return ret;
}


void InPlaceParser::DefaultSymbols(void)
{
	SetHardSeparator(',');
	SetHardSeparator('(');
	SetHardSeparator(')');
	SetHardSeparator('=');
	SetHardSeparator('[');
	SetHardSeparator(']');
	SetHardSeparator('{');
	SetHardSeparator('}');
	SetCommentSymbol('#');
}


const char ** InPlaceParser::GetArglist(char *line,int &count) // convert source string into an arg list, this is a destructive parse.
{
	const char **ret = 0;

	static const char *argv[MAXARGS];
	int argc = 0;

	char *foo = line;

	while ( !EOS(*foo) && argc < MAXARGS )
	{

		foo = SkipSpaces(foo); // skip any leading spaces

		if ( EOS(*foo) ) break;

		if ( *foo == mQuoteChar ) // if it is an open quote
		{
			foo++;
			if ( argc < MAXARGS )
			{
				argv[argc++] = foo;
			}
			while ( !EOS(*foo) && *foo != mQuoteChar ) foo++;
			if ( !EOS(*foo) )
			{
				*foo = 0; // replace close quote with zero byte EOS
				foo++;
			}
		}
		else
		{

			foo = AddHard(argc,argv,foo); // add any hard separators, skip any spaces

			if ( IsNonSeparator(*foo) )  // add non-hard argument.
			{
				bool quote  = false;
				if ( *foo == mQuoteChar )
				{
					foo++;
					quote = true;
				}

				if ( argc < MAXARGS )
				{
					argv[argc++] = foo;
				}

				if ( quote )
				{
					while (*foo && *foo != mQuoteChar ) foo++;
					if ( *foo ) *foo = 32;
				}

				// continue..until we hit an eos ..
				while ( !EOS(*foo) ) // until we hit EOS
				{
					if ( IsWhiteSpace(*foo) ) // if we hit a space, stomp a zero byte, and exit
					{
						*foo = 0;
						foo++;
						break;
					}
					else if ( IsHard(*foo) ) // if we hit a hard separator, stomp a zero byte and store the hard separator argument
					{
						const char *hard = &mHardString[*foo*2];
						*foo = 0;
						if ( argc < MAXARGS )
						{
							argv[argc++] = hard;
						}
						foo++;
						break;
					}
					foo++;
				} // end of while loop...
			}
		}
	}

	count = argc;
	if ( argc )
	{
		ret = argv;
	}

	return ret;
}

/*******************************************************************/
/******************** Geometry.h  ********************************/
/*******************************************************************/

class GeometryVertex
{
public:
	float        mPos[3];
	float        mNormal[3];
	float        mTexel[2];
};


class GeometryInterface
{
public:

	virtual void NodeTriangle(const GeometryVertex *v1,const GeometryVertex *v2,const GeometryVertex *v3, bool textured)
	{
	}

};


/*******************************************************************/
/******************** Obj.h  ********************************/
/*******************************************************************/


class OBJ : public InPlaceParserInterface
{
public:
  int LoadMesh(const char *fname,GeometryInterface *callback, bool textured);
  int ParseLine(int lineno,int argc,const char **argv);  // return TRUE to continue parsing, return FALSE to abort parsing process
private:

  void GetVertex(GeometryVertex &v,const char *face) const;

  FloatVector     mVerts;
  FloatVector     mTexels;
  FloatVector     mNormals;

  bool            mTextured;

  GeometryInterface *mCallback;
};


/*******************************************************************/
/******************** Obj.cpp  ********************************/
/*******************************************************************/

__forceinline float parseFloat(const char* p)
{
	int s = 1;
	while (*p == ' ') p++;

	if (*p == '-') {
		s = -1; p++;
	}

	float acc = 0;
	while (*p >= '0' && *p <= '9')
		acc = acc * 10 + *p++ - '0';

	if (*p == '.') {
		float k = 0.1f;
		p++;
		while (*p >= '0' && *p <= '9') {
			acc += (*p++ - '0') * k;
			k *= 0.1f;
		}
	}

	return s * acc;
}


int OBJ::LoadMesh(const char *fname,GeometryInterface *iface, bool textured)
{
  mTextured = textured;
  int ret = 0;

  mVerts.clear();
  mTexels.clear();
  mNormals.clear();

  mCallback = iface;

  InPlaceParser ipp(fname);

  ipp.Parse(this);

return ret;
}

void OBJ::GetVertex(GeometryVertex &v,const char *face) const
{
  v.mPos[0] = 0;
  v.mPos[1] = 0;
  v.mPos[2] = 0;

  v.mTexel[0] = 0;
  v.mTexel[1] = 0;

  v.mNormal[0] = 0;
  v.mNormal[1] = 1;
  v.mNormal[2] = 0;

  int index = atoi( face )-1;

  const char *texel = strstr(face,"/");

  if ( texel )
  {
    int tindex = atoi( texel+1) - 1;

    if ( tindex >=0 && tindex < (int)(mTexels.size()/2) )
    {
    	const float *t = &mTexels[tindex*2];

      v.mTexel[0] = t[0];
      v.mTexel[1] = t[1];

    }

    const char *normal = strstr(texel+1,"/");
    if ( normal )
    {
      int nindex = atoi( normal+1 ) - 1;

      if (nindex >= 0 && nindex < (int)(mNormals.size()/3) )
      {
      	const float *n = &mNormals[nindex*3];

        v.mNormal[0] = n[0];
        v.mNormal[1] = n[1];
        v.mNormal[2] = n[2];
      }
    }
  }

  if ( index >= 0 && index < (int)(mVerts.size()/3) )
  {

    const float *p = &mVerts[index*3];

    v.mPos[0] = p[0];
    v.mPos[1] = p[1];
    v.mPos[2] = p[2];
  }

}

int OBJ::ParseLine(int lineno,int argc,const char **argv)  // return TRUE to continue parsing, return FALSE to abort parsing process
{
	int ret = 0;

	if ( argc >= 1 )
	{
		const char *foo = argv[0];
		if ( *foo != '#' )
		{
			if ( _stricmp(argv[0],"v") == 0 && argc == 4 )
			{
				//float vx = (float) atof( argv[1] );
				//float vy = (float) atof( argv[2] );
				//float vz = (float) atof( argv[3] );
				float vx, vy, vz;
				vx = parseFloat(argv[1]);
				vy = parseFloat(argv[2]);
				vz = parseFloat(argv[3]);
				mVerts.push_back(vx);
				mVerts.push_back(vy);
				mVerts.push_back(vz);
			}
			else if ( _stricmp(argv[0],"vt") == 0 && (argc == 3 || argc == 4))
			{
				// ignore 4rd component if present
				//float tx = atof( argv[1] );
				//float ty = atof( argv[2] );
				float tx, ty;
				tx = parseFloat(argv[1]);
				ty = parseFloat(argv[2]);
				mTexels.push_back(tx);
				mTexels.push_back(ty);
			}
			else if ( _stricmp(argv[0],"vn") == 0 && argc == 4 )
			{
				//float normalx = atof(argv[1]);
				//float normaly = atof(argv[2]);
				//float normalz = atof(argv[3]);
				float normalx, normaly, normalz;
				normalx = parseFloat(argv[1]);
				normaly = parseFloat(argv[2]);
				normalz = parseFloat(argv[3]);
				mNormals.push_back(normalx);
				mNormals.push_back(normaly);
				mNormals.push_back(normalz);
			}
			else if ( _stricmp(argv[0],"f") == 0 && argc >= 4 )
			{
				GeometryVertex v[32];

				int vcount = argc-1;

				for (int i=1; i<argc; i++)
				{
					GetVertex(v[i-1],argv[i] );
				}

				mCallback->NodeTriangle(&v[0],&v[1],&v[2], mTextured);

				if ( vcount >=3 ) // do the fan
				{
					for (int i=2; i<(vcount-1); i++)
					{
						mCallback->NodeTriangle(&v[0],&v[i],&v[i+1], mTextured);
					}
				}
			}
		}
	}

	return ret;
}




class BuildMesh : public GeometryInterface
{
public:

	int GetIndex(const float *p, const float *n, const float *texCoord)
	{

		int vcount = (int)mVertices.size()/3;

		if(vcount>0)
		{
			//New MS STL library checks indices in debug build, so zero causes an assert if it is empty.
			const float *v = &mVertices[0];
			const float *norm = &mNormals[0];
			const float *t = texCoord != NULL ? &mTexCoords[0] : NULL;

			for (int i=0; i<vcount; i++)
			{
				if ( v[0] == p[0] && v[1] == p[1] && v[2] == p[2] )
				{
					if ( norm[0] == n[0] && norm[1] == n[1] && norm[2] == n[2] )
					{
						if (texCoord == NULL || (t[0] == texCoord[0] && t[1] == 1.0f-texCoord[1]))
						{
							return i;
						}
					}
				}
				v+=3;
				if (t != NULL)
					t += 2;
			}
		}

		mVertices.push_back( p[0] );
		mVertices.push_back( p[1] );
		mVertices.push_back( p[2] );

		mNormals.push_back( n[0] );
		mNormals.push_back( n[1] );
		mNormals.push_back( n[2] );

		if (texCoord != NULL)
		{
			mTexCoords.push_back( texCoord[0] );
			mTexCoords.push_back( 1.0f-texCoord[1] );
		}

		return vcount;
	}

	virtual void NodeTriangle(const GeometryVertex *v1,const GeometryVertex *v2,const GeometryVertex *v3, bool textured)
	{
		mIndices.push_back( GetIndex(v1->mPos, v1->mNormal, textured ? v1->mTexel : NULL) );
		mIndices.push_back( GetIndex(v2->mPos, v2->mNormal, textured ? v2->mTexel : NULL) );
		mIndices.push_back( GetIndex(v3->mPos, v3->mNormal, textured ? v3->mTexel : NULL) );
	}

  const FloatVector& GetVertices(void) const { return mVertices; };
  const FloatVector& GetNormals(void) const { return mNormals; };
  const FloatVector& GetTexCoords(void) const { return mTexCoords; };
  const IntVector& GetIndices(void) const { return mIndices; };

private:
  FloatVector     mVertices;
  FloatVector     mNormals;
  FloatVector     mTexCoords;
  IntVector       mIndices;
};

OBJLoader::OBJLoader(void)
{
	mVertexCount = 0;
	mTriCount    = 0;

	mIndices     = NULL;
	mVertices    = NULL;
	mNormals	 = NULL;
	mTexCoords   = NULL;
}

OBJLoader::~OBJLoader(void)
{
	Exit();
}

void OBJLoader::Exit(void)
{
	if (mVertices != NULL)
	{
		delete mVertices;
		mVertices = NULL;
	}

	if (mNormals != NULL)
	{
		delete mNormals;
		mNormals = NULL;
	}

	if (mTexCoords != NULL)
	{
		delete mTexCoords;
		mTexCoords = NULL;
	}

	if (mIndices != NULL)
	{
		delete mIndices;
		mIndices = 0;
	}
}

unsigned int OBJLoader::LoadObj(const char *fname, bool textured)
{

	unsigned int ret = 0;

	Exit();

	mVertexCount = 0;
	mTriCount = 0;

	BuildMesh bm;
	
	OBJ obj;
	obj.LoadMesh(fname, &bm, textured);

	const FloatVector &vlist = bm.GetVertices();
	const FloatVector &nlist = bm.GetNormals();
	const IntVector &indices = bm.GetIndices();
	if ( vlist.size() )
	{
		mVertexCount = (int)vlist.size()/3;
		mVertices = new float[mVertexCount*3];
		memcpy( mVertices, &vlist[0], sizeof(float)*mVertexCount*3 );

		mNormals = new float[mVertexCount*3];
		memcpy( mNormals, &nlist[0], sizeof(float)*mVertexCount*3 );

		if (textured)
		{
			mTexCoords = new float[mVertexCount * 2];
			const FloatVector& tList = bm.GetTexCoords();
			memcpy( mTexCoords, &tList[0], sizeof(float) * mVertexCount * 2);
		}

		mTriCount = (int)indices.size()/3;
		mIndices = new int[mTriCount*3*sizeof(int)];
		memcpy(mIndices, &indices[0], sizeof(int)*mTriCount*3);
		ret = mTriCount;
	}

	return ret;
}

