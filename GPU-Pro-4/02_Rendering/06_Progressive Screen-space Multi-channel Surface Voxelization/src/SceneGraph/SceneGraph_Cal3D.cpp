//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//  Scene Graph 3D                                                          //
//  Athanasios Gaitatzes (gaitat at yahoo dot com), 2009                    //
//                                                                          //
//  This is a free, extensible scene graph management library that works    //
//  along with the EaZD deferred renderer. Both libraries and their source  //
//  code are free. If you use this code as is or any part of it in any kind //
//  of project or product, please acknowledge the source and its author.    //
//                                                                          //
//  For manuals, help and instructions, please visit:                       //
//  http://graphics.cs.aueb.gr/graphics/                                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <float.h>

#include "SceneGraph.h"
#include "Texture2D.h"

#define DR_CAL3D_UPDATE_CYCLE 2
int Cal3D::new_update_offset_ = 0;

Cal3D::Cal3D (void)
{
	calModel_ = NULL;
	calCoreModel_ = NULL;

	totalCycles_ = 0;
	totalActions_ = 0;

	pause_ = false;
	last_time_ = 0.f;

	alpha_ = false;

	update_offset_ = new_update_offset_;
	new_update_offset_ = (new_update_offset_+1)%DR_CAL3D_UPDATE_CYCLE;
	update_ = 0;
}

Cal3D::~Cal3D (void)
{
	free (path_);

	if (calModel_)      delete calModel_;       // destroy model instance
	if (calCoreModel_)  delete calCoreModel_;   // destroy core model instance

	if (meshVertices_)              delete [] meshVertices_;
	if (meshNormals_)               delete [] meshNormals_;
	if (meshTextureCoordinates_)    delete [] meshTextureCoordinates_;
	if (meshFaces_)                 delete [] meshFaces_;
}

void Cal3D::parse (xmlNodePtr pXMLNode)
{
	char * val = NULL;

	val = (char *)xmlGetProp (pXMLNode, (xmlChar *)"file");
	if (val)
	{
		setName (val);

		path_ = world->getFullPath (val);
		if (path_)
			load (path_);
		else
			EAZD_TRACE ("Cal3D::parse() : ERROR - File \"" << val << "\" is corrupt or does not exist.");

		xmlFree (val);
	}

	val = (char *)xmlGetProp (pXMLNode, (xmlChar *)"alpha");
	if (val)
	{
		parseBoolean (alpha_, val);

		xmlFree (val);
	}

	Node3D::parse (pXMLNode);
}

void Cal3D::load (const string & path)
{
	int line, index = 0;

	string cDir, currentDir;
	cDir.assign (path);
	currentDir.assign (cDir.substr (0, cDir.find_last_of ("/\\")+1));

	// open the model configuration file
	ifstream file;
	file.open (path.c_str (), ios::in | ios::binary);
	if (! file)
	{
		EAZD_TRACE ("Cal3D::load() : ERROR - Failed to open model configuration file \"" << path << "\".");
		exit (EXIT_FAILURE);
	}

	string nm = "dummy";
	// create a core model instance
	calCoreModel_ = new CalCoreModel(nm);

	// parse all lines from the model configuration file
	for (line = 1;; line++)
	{
		// read the next model configuration line
		string lineBuf;
		getline (file, lineBuf);

		// stop if we reached the end of file
		if (file.eof ())
			break;

		// check if an error happend while reading from the file
		if (! file)
		{
			EAZD_TRACE ("Cal3D::load() : ERROR - Error while reading from the model configuration file \"" << path << "\".");
			exit (EXIT_FAILURE);
		}

		// find the first non-whitespace character
		string::size_type pos;
		pos = lineBuf.find_first_not_of (" \t");

		// check for empty lines
		if ((pos == string::npos) || (lineBuf[pos] == '\n') ||
			(lineBuf[pos] == '\r') || (lineBuf[pos] == 0))
			continue;

		// check for comment lines
		if (lineBuf[pos] == '#')
			continue;

		// get the key
		string strKey;
		strKey = lineBuf.substr (pos, lineBuf.find_first_of (" =\t\n\r", pos) - pos);
		pos += strKey.size ();

		// get the '=' character
		pos = lineBuf.find_first_not_of (" \t", pos);
		if ((pos == string::npos) || (lineBuf[pos] != '='))
		{
			EAZD_TRACE ("Cal3D::load() : ERROR - File \"" << path << "\" (" << line << "): Invalid syntax.");
			exit (EXIT_FAILURE);
		}

		// find the first non-whitespace character after the '=' character
		pos = lineBuf.find_first_not_of (" \t", pos + 1);

		// get the data
		string strData = lineBuf.substr (pos, lineBuf.find_first_of ("\n\r", pos) - pos);

CalLoader::setLoadingMode (LOADER_INVERT_V_COORD);

		// handle the model creation
			 if (strKey == "scale")
			;
		else if (strKey == "fliptexture")
		{
			if (strData == "on" || strData == "true")
			{
				EAZD_PRINT ("Cal3D::load() : INFO - Flipping UV Coordinates");
				CalLoader::setLoadingMode (LOADER_INVERT_V_COORD);
			}
		}
		else if (strKey == "skeleton")
		{
			// load core skeleton
		//	EAZD_PRINT ("Cal3D::load() : INFO - Loading skeleton \"" << strData << "\"");
			if (! calCoreModel_->loadCoreSkeleton (currentDir+strData))
			{
				CalError::printLastError ();
				exit (EXIT_FAILURE);
			}
		}
		else if (strKey == "animation")
		{
			// load core animation
		//	EAZD_PRINT ("Cal3D::load() : INFO - Loading animation \"" << strData << "\"");
			animationCycle_[totalCycles_++] = index++;
			if (calCoreModel_->loadCoreAnimation (currentDir+strData) == -1)
			{
				CalError::printLastError ();
				exit (EXIT_FAILURE);
			}
		}
		else if (strKey == "action")
		{
			// load core animation
		//	EAZD_PRINT ("Cal3D::load() : INFO - Loading animation \"" << strData << "\"");
			animationAction_[totalActions_++] = index++;
			if (calCoreModel_->loadCoreAnimation (currentDir+strData) == -1)
			{
				CalError::printLastError ();
				exit (EXIT_FAILURE);
			}
		}
		else if (strKey == "mesh")
		{
			// load core mesh
			EAZD_PRINT ("Cal3D::load() : INFO - Loading mesh \"" << strData << "\"");
			if (calCoreModel_->loadCoreMesh (currentDir+strData) == -1)
			{
				CalError::printLastError ();
				exit (EXIT_FAILURE);
			}
		}
		else if (strKey == "material")
		{
			// load core material
		//	EAZD_PRINT ("Cal3D::load() : INFO - Loading material \"" << strData << "\"");
			if (calCoreModel_->loadCoreMaterial (currentDir+strData) == -1)
			{
				CalError::printLastError ();
				exit (EXIT_FAILURE);
			}
		}
		else
		{
			// everything else triggers an error message, but is ignored
			EAZD_PRINT (path << "(" << line << "): Invalid syntax.");
		}
	}

	// explicitely close the file
	file.close ();
}

// Load and create a texture from a given file
GLuint loadTexture (const string & path)
{
	GLuint  textureId = 0;

	if (STR_EQUAL (strrchr (path.c_str (), '.'), ".raw"))
	{
		// open the texture file
		ifstream file;
		file.open (path.c_str (), ios::in | ios::binary);
		if (!file)
		{
			EAZD_TRACE ("Cal3D::loadTexture() : ERROR - Texture file \"" << path << "\" not found.");
			exit (EXIT_FAILURE);
		}

		// load the dimension of the texture
		int width;  file.read ((char *) &width,  4);
		int height; file.read ((char *) &height, 4);
		int depth;  file.read ((char *) &depth,  4);

		// allocate a temporary buffer to load the texture to
		unsigned char *pixels = new unsigned char[2 * width * height * depth];
		if (! pixels)
		{
			EAZD_TRACE ("Cal3D::loadTexture() : ERROR - Memory allocation for texture \"" << path << "\" failed.");
			exit (EXIT_FAILURE);
		}

		file.read ((char *) pixels, width * height * depth); // load the texture
		file.close ();          // explicitly close the file

		// flip texture around y-axis (-> opengl-style)
		for (int y = 0; y < height; y++)
		{
			memcpy (&pixels[(height + y)     * width * depth],
					&pixels[(height - y - 1) * width * depth], width * depth);
		}

		// generate texture
		glPixelStorei (GL_UNPACK_ALIGNMENT, 1);

		if (glIsTexture (textureId))
			glDeleteTextures (1, &textureId);
		glGenTextures (1, &textureId);
		glBindTexture (GL_TEXTURE_2D, textureId);
		glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // GL_LINEAR);

	 // glTexImage2D (GL_TEXTURE_2D, 0, (depth == 3) ? GL_RGB : GL_RGBA,
	 //               width, height, 0, (depth == 3) ? GL_RGB : GL_RGBA,
	 //               GL_UNSIGNED_BYTE, &pixels[width * height * depth]);
		gluBuild2DMipmaps (GL_TEXTURE_2D, (depth == 3) ? GL_RGB : GL_RGBA,
						   width, height, (depth == 3) ? GL_RGB : GL_RGBA,
						   GL_UNSIGNED_BYTE, &pixels[width * height * depth]);

		// free the allocated memory
		delete[] pixels;
	}
	else
	{
		Texture2D *tex = new Texture2D ((char *) path.c_str ());

		textureId = tex->getID ();
	}

	return textureId;
}

void Cal3D::createDataArrays (void)
{
	int maxVertices = 0;
	int maxFaces = 0;

	// get the renderer of the model
	CalRenderer *pCalRenderer = calModel_->getRenderer ();

	// begin the rendering loop
	if (pCalRenderer->beginRendering ())
	{
		// get the number of meshes
		int     meshCount = pCalRenderer->getMeshCount ();

		// render all meshes of the model
		for (int meshId = 0; meshId < meshCount; meshId++)
		{
			// get the number of submeshes
			int     submeshCount = pCalRenderer->getSubmeshCount (meshId);

			// render all submeshes of the mesh
			for (int submeshId = 0; submeshId < submeshCount; submeshId++)
			{
				// select mesh and submesh for further data access
				if (pCalRenderer->selectMeshSubmesh (meshId, submeshId))
				{
					maxVertices = MAX2 (maxVertices, pCalRenderer->getVertexCount ());
					maxFaces    = MAX2 (maxFaces,    pCalRenderer->getFaceCount ());
				}
			}
		}

		// end the rendering
		pCalRenderer->endRendering ();

		meshVertices_           = new float[maxVertices][3];
		meshNormals_            = new float[maxVertices][3];
		meshTextureCoordinates_ = new float[maxVertices][2];
		meshFaces_              = new CalIndex[maxFaces][3];
	}

	return;
}

void Cal3D::init (void)
{
	// get the directory of the texture
	string cDir, currentDir;
	cDir.assign (path_);
	currentDir.assign (cDir.substr (0, cDir.find_last_of ("/\\")+1));

	// make one material thread for each material
	int     materialId;

	for (materialId = 0;
		 materialId < calCoreModel_->getCoreMaterialCount ();
		 materialId++)
	{
		// create the a material thread
		calCoreModel_->createCoreMaterialThread (materialId);

		// initialize the material thread
		calCoreModel_->setCoreMaterialId (materialId, 0, materialId);
	}

	// needs to be done in two steps
	for (materialId = 0;
		 materialId < calCoreModel_->getCoreMaterialCount ();
		 materialId++)
	{
		// get the core material
		CalCoreMaterial *pCoreMaterial = calCoreModel_->getCoreMaterial (materialId);

		// loop through all maps of the core material
		for (int mapId = 0; mapId < pCoreMaterial->getMapCount (); mapId++)
		{
		//	EAZD_PRINT ("Cal3D::init() : INFO - Loading texture \"" <<
		//		pCoreMaterial->getMapFilename (mapId) << "\"");

			// load the texture from the file and
			GLuint  textureId = loadTexture (currentDir + pCoreMaterial->getMapFilename (mapId));

			// store the opengl texture id in the user data of the map
			pCoreMaterial->setMapUserData (mapId, (Cal::UserData) textureId);
				// (Cal::UserData) loadTexture (currentDir + pCoreMaterial->getMapFilename (mapId)));
		}
	}

	// create the model instance from the loaded core model
	calModel_ = new CalModel (calCoreModel_);

	// attach all meshes to the model
	for (int meshId = 0; meshId < calCoreModel_->getCoreMeshCount (); meshId++)
		calModel_->attachMesh (meshId);

	// set the material set of the whole model
	calModel_->setMaterialSet (0);

	// Initialize Bounding Box calculation
	if (getDebug ())
		calCoreModel_->getCoreSkeleton ()->calculateBoundingBoxes (calCoreModel_);

	// Initialize to start with Cycle 0
	calModel_->getMixer ()->blendCycle (animationCycle_[0], 1.0f, 0.0f);
	previousCycle_ = 0;

	calModel_->update (0.f);

	createDataArrays ();    // create the space for the mesh data
	calcBSphere ();         // compute the bounding sphere

	last_time_ = world->getTime () * 1000.0;

	Node3D::init ();
}

void Cal3D::app (void)
{
	// calculate the amount of elapsed seconds
	float   elapsedTime = (float) world->getTime () - last_time_ / 1000.f;
	
	if (update_== update_offset_)
	{
		if (! pause_)
			calModel_->update (elapsedTime);

		last_time_ = world->getTime () * 1000.0;
	}
	update_ = update_offset_;//(update_+1)%DR_CAL3D_UPDATE_CYCLE;
}

void Cal3D::calcBSphere (void)
{
	Vector3D min = Vector3D (FLT_MAX, FLT_MAX, FLT_MAX);
	Vector3D max = Vector3D (FLT_MIN, FLT_MIN, FLT_MIN);

	// get the renderer of the model
	CalRenderer *pCalRenderer = calModel_->getRenderer ();

	// begin the rendering loop
	if (pCalRenderer->beginRendering ())
	{
		// get the number of meshes
		int     meshCount = pCalRenderer->getMeshCount ();

		// render all meshes of the model
		for (int meshId = 0; meshId < meshCount; meshId++)
		{
			// get the number of submeshes
			int     submeshCount = pCalRenderer->getSubmeshCount (meshId);

			for (int submeshId = 0; submeshId < submeshCount; submeshId++)
			{
				// select mesh and submesh for further data access
				if (pCalRenderer->selectMeshSubmesh (meshId, submeshId))
				{
					// get the transformed vertices of the submesh
					int vertexCount = pCalRenderer->getVertices (&meshVertices_[0][0]);

					// get the faces of the submesh
					int faceCount = pCalRenderer->getFaces (&meshFaces_[0][0]);

					for (int i = 0; i < faceCount; i++)
					for (int face = 0; face < 3; face++)
					{
						int f = meshFaces_[i][face];

						// Compare with min/max
						min.x = MIN2 (min.x, meshVertices_[f][0]);
						max.x = MAX2 (max.x, meshVertices_[f][0]);

						min.y = MIN2 (min.y, meshVertices_[f][1]);
						max.y = MAX2 (max.y, meshVertices_[f][1]);

						min.z = MIN2 (min.z, meshVertices_[f][2]);
						max.z = MAX2 (max.z, meshVertices_[f][2]);
					}
				} // select mesh and submesh for further data access
			}
		} // render all meshes of the model

		// end the rendering
		pCalRenderer->endRendering ();
	}

	bsphere_.set ((min + max) * 0.5f,
				  0.5f * (max - min).length ());
}

void Cal3D::drawSkeleton (void)
{
	// draw the bone lines
	float lines[1024][2][3];
	int nrLines =  calModel_->getSkeleton ()->getBoneLines (&lines[0][0][0]);

	glLineWidth (3.0f);
	glColor3f (0.0f, 1.0f, 0.0f);
	glBegin (GL_LINES);
	for (int currLine = 0; currLine < nrLines; currLine++)
	{
		glVertex3f (lines[currLine][0][0],
					lines[currLine][0][1],
					lines[currLine][0][2]);
		glVertex3f (lines[currLine][1][0],
					lines[currLine][1][1],
					lines[currLine][1][2]);
	}
	glEnd ();
	glLineWidth (1.0f);

	// draw the bone points
	float points[1024][3];
	int nrPoints = calModel_->getSkeleton ()->getBonePoints (&points[0][0]);

	glPointSize (4.0f);
	glColor3f (1.0f, 1.0f, 0.0f);
	glBegin (GL_POINTS);
	for (int currPoint = 0; currPoint < nrPoints; currPoint++)
	{
		glVertex3f (points[currPoint][0],
					points[currPoint][1],
					points[currPoint][2]);
	}
	glEnd ();
	glPointSize (1.0f);
} // end of drawSkeleton

void Cal3D::drawBBox (void)
{
	glColor3f (1.0f, 1.0f, 0.0f);

	calModel_->getSkeleton ()->calculateBoundingBoxes ();
	CalSkeleton * pSkeleton = calModel_->getSkeleton ();
	vector <CalBone *> vectorBone = pSkeleton->getVectorBone ();

	glBegin (GL_LINES);
	for (unsigned int boneId = 0; boneId < vectorBone.size (); boneId++)
	{
		CalVector p[8];
		vectorBone[boneId]->getBoundingBox ().computePoints (p);

		glVertex3f (p[0].x, p[0].y, p[0].z);
		glVertex3f (p[1].x, p[1].y, p[1].z);

		glVertex3f (p[4].x, p[4].y, p[4].z);
		glVertex3f (p[5].x, p[5].y, p[5].z);

		glVertex3f (p[0].x, p[0].y, p[0].z);
		glVertex3f (p[4].x, p[4].y, p[4].z);

		glVertex3f (p[5].x, p[5].y, p[5].z);
		glVertex3f (p[1].x, p[1].y, p[1].z);

		glVertex3f (p[0].x, p[0].y, p[0].z);
		glVertex3f (p[2].x, p[2].y, p[2].z);

		glVertex3f (p[3].x, p[3].y, p[3].z);
		glVertex3f (p[1].x, p[1].y, p[1].z);

		glVertex3f (p[4].x, p[4].y, p[4].z);
		glVertex3f (p[6].x, p[6].y, p[6].z);

		glVertex3f (p[5].x, p[5].y, p[5].z);
		glVertex3f (p[7].x, p[7].y, p[7].z);

		glVertex3f (p[2].x, p[2].y, p[2].z);
		glVertex3f (p[6].x, p[6].y, p[6].z);

		glVertex3f (p[2].x, p[2].y, p[2].z);
		glVertex3f (p[3].x, p[3].y, p[3].z);

		glVertex3f (p[3].x, p[3].y, p[3].z);
		glVertex3f (p[7].x, p[7].y, p[7].z);

		glVertex3f (p[6].x, p[6].y, p[6].z);
		glVertex3f (p[7].x, p[7].y, p[7].z);
	}
	glEnd ();
} // end of drawBBox

void Cal3D::drawGrid ()
{
	glColor3f (0.3f, 0.3f, 0.3f);
	glBegin (GL_LINES);

	// Draw grid.
	for (float i = -100.0f; i <= 100.0f; i += 10.0f)
	{
		glVertex2f (-100.0f, i);
		glVertex2f ( 100.0f, i);

		glVertex2f (i, -100.0f);
		glVertex2f (i,  100.0f);
	}

	// Draw axis lines.
	glColor3f  (1, 0, 0);
	glVertex3f (0, 0, 0);
	glVertex3f (100, 0, 0);

	glColor3f  (0, 1, 0);
	glVertex3f (0, 0, 0);
	glVertex3f (0, 100, 0);

	glColor3f  (0, 0, 1);
	glVertex3f (0, 0, 0);
	glVertex3f (0, 0, 100);

	glEnd ();
}

void Cal3D::draw (void)
{
	int   matrixMode;

	if (render_mode != SCENE_GRAPH_RENDER_MODE_NORMAL)
		return;

	// save current matrix mode
	glGetIntegerv (GL_MATRIX_MODE, &matrixMode);
	glMatrixMode (GL_MODELVIEW);
	glPushMatrix ();
	glRotatef (-90.f, 1.f, 0.f, 0.f);   // to bring the model upright

	if (getDebug ())
	{
		glDisable (GL_DEPTH_TEST);
		glDisable (GL_LIGHTING);
		glEnable  (GL_COLOR_MATERIAL);      // glColor has priority

		bsphere_.draw ();
		drawSkeleton ();
		drawBBox ();
	 // drawGrid ();

		glDisable(GL_COLOR_MATERIAL);
		glEnable (GL_LIGHTING);
		glEnable (GL_DEPTH_TEST);
	}

	int totalFaceCount=0, totalVertexCount=0;

	// get the renderer of the model
	CalRenderer *pCalRenderer = calModel_->getRenderer ();

	// begin the rendering loop
	if (pCalRenderer->beginRendering ())
	{
		glEnable (GL_NORMALIZE);
		glDisable (GL_COLOR_MATERIAL);       // glMaterial has priority

		// we will use vertex arrays, so enable them
		glEnableClientState (GL_VERTEX_ARRAY);
		glEnableClientState (GL_NORMAL_ARRAY);

		// get the number of meshes
		int     meshCount = pCalRenderer->getMeshCount ();

		// render all meshes of the model
		for (int meshId = 0; meshId < meshCount; meshId++)
		{
			// get the number of submeshes
			int     submeshCount = pCalRenderer->getSubmeshCount (meshId);

			// render all submeshes of the mesh
			for (int submeshId = 0; submeshId < submeshCount; submeshId++)
			{
				// select mesh and submesh for further data access
				if (pCalRenderer->selectMeshSubmesh (meshId, submeshId))
				{
					unsigned char meshColor[4];
					GLfloat materialColor[4];

					// set the material ambient color
					pCalRenderer->getAmbientColor (&meshColor[0]);
					materialColor[0] = meshColor[0] / 255.0f;
					materialColor[1] = meshColor[1] / 255.0f;
					materialColor[2] = meshColor[2] / 255.0f;
					materialColor[3] = alpha_ ? meshColor[3] / 255.0f : 1.0f;
					glMaterialfv (GL_FRONT, GL_AMBIENT, materialColor);

					// set the material diffuse color
					pCalRenderer->getDiffuseColor (&meshColor[0]);
					materialColor[0] = meshColor[0] / 255.0f;
					materialColor[1] = meshColor[1] / 255.0f;
					materialColor[2] = meshColor[2] / 255.0f;
					materialColor[3] = alpha_ ? meshColor[3] / 255.0f : 1.0f;
					glMaterialfv (GL_FRONT, GL_DIFFUSE, materialColor);

					// set the material specular color
					pCalRenderer->getSpecularColor (&meshColor[0]);
					materialColor[0] = meshColor[0] / 255.0f;
					materialColor[1] = meshColor[1] / 255.0f;
					materialColor[2] = meshColor[2] / 255.0f;
					materialColor[3] = alpha_ ? meshColor[3] / 255.0f : 1.0f;
					glMaterialfv (GL_FRONT, GL_SPECULAR, materialColor);

					// set the material shininess factor
					float shininess = (1.f - pCalRenderer->getShininess ()) * 128.f;
					glMaterialfv (GL_FRONT, GL_SHININESS, &shininess);

					// get the transformed vertices of the submesh
					int vertexCount = pCalRenderer->getVertices (&meshVertices_[0][0]);
					totalVertexCount += vertexCount;

					// get the transformed normals of the submesh
					pCalRenderer->getNormals (&meshNormals_[0][0]);

					// get the texture coordinates of the submesh
					int textureCoordinateCount = pCalRenderer->getTextureCoordinates (0, &meshTextureCoordinates_[0][0]);

					// get the faces of the submesh
					int faceCount = pCalRenderer->getFaces (&meshFaces_[0][0]);
					totalFaceCount += faceCount;

					// set the vertex and normal buffers
					glVertexPointer (3, GL_FLOAT, 0, &meshVertices_[0][0]);
					glNormalPointer (GL_FLOAT, 0, &meshNormals_[0][0]);

					// set the texture coordinate buffer and state if necessary
					if (pCalRenderer->getMapCount () > 0 &&
						textureCoordinateCount > 0)
					{
						glEnable (GL_TEXTURE_2D);
						glEnableClientState (GL_TEXTURE_COORD_ARRAY);

						// set the texture id we stored in the map user data
						glActiveTexture(GL_TEXTURE0);
						glBindTexture (GL_TEXTURE_2D,
									(GLuint) pCalRenderer->getMapUserData (0));

						// set the texture coordinate buffer
						glTexCoordPointer (2, GL_FLOAT, 0, &meshTextureCoordinates_[0][0]);
					}

					// Draw the submesh
					glDrawElements (GL_TRIANGLES, faceCount * 3,
									sizeof (CalIndex) == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT,
									&meshFaces_[0][0]);

#if 0
					glBegin (GL_TRIANGLES);
						for (int i = 0; i < faceCount; i++)
						for (int face = 0; face < 3; face++)
						{
							int f = meshFaces_[i][face];

							glNormal3f (meshNormals_[f][0],
										meshNormals_[f][1],
										meshNormals_[f][2]);
							if (pCalRenderer->getMapCount () > 0 &&
								textureCoordinateCount > 0)
							{
								glTexCoord2f (
									meshTextureCoordinates_[f][0],
									meshTextureCoordinates_[f][1]);
							}
							glVertex3f (meshVertices_[f][0],
										meshVertices_[f][1],
										meshVertices_[f][2]);
						}
					glEnd ();
#endif

					// disable the texture coordinate state if necessary
					if (pCalRenderer->getMapCount () > 0 &&
						textureCoordinateCount > 0)
					{
						glDisableClientState (GL_TEXTURE_COORD_ARRAY);

						glDisable (GL_TEXTURE_2D);
						glBindTexture (GL_TEXTURE_2D, 0);
					}
				}
			}
		}

		// clear vertex array state
		glDisableClientState (GL_NORMAL_ARRAY);
		glDisableClientState (GL_VERTEX_ARRAY);

		// end the rendering
		pCalRenderer->endRendering ();
	}

	glPopMatrix ();

	// restore user's matrix mode
	glMatrixMode (matrixMode);
}

