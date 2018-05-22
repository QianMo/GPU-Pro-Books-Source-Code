
#include "GlobalIlluminationRenderer_IV.h"
#include "shaders/DeferredRendererShader_GI_IV.h"
#include "SceneGraph.h"

#define MAP_MINUS1TO1_0TO1(_value)  (0.5f * (_value) + 0.5f)	// Map [-1.0,1.0] to [0.0,1.0]
#define MAP_0TO1_MINUS1TO1(_value)  (2.0f * (_value) - 1.0f)	// Map [0.0,1.0] to [-1.0,1.0]

#define DO_TIMINGS

#define TEX_BITS	GL_RGBA16F	// GL_R16F
#define TEX_FORMAT	GL_RGBA		// GL_LUMINANCE
#define TEX_TYPE	GL_HALF_FLOAT
#define TEX_BANDS	4

GlobalIlluminationRendererIV::GlobalIlluminationRendererIV()
{
	injection_SS_shader = new DRShaderGI_IV_Injection_SS();
	injection_SS_Cleanup_shader = new DRShaderGI_IV_Injection_SS_Cleanup();
	propagation_shader = new DRShaderGI_IV_Propagation();

	if (shader) delete shader;
	shader = new DRShaderGI_IV();

	type = DR_GI_METHOD_IV;
	resolution = 16;
	m_format = TEX_FORMAT;
	params_decoded = false;
	inj_width = 128;
	inj_height = 128;
	fbVerts = NULL;
    write_debug_data = false;

	inj_camera = true, inj_lights = true;

	propagateFboId[0] = propagateFboId[1] = 0;
	propagateTexId[0] = propagateTexId[1] = propagateTexId[2] = 0;
	propagateTexId[3] = propagateTexId[4] = propagateTexId[5] = 0;
    inject_SSFboId[0] = inject_SSFboId[1] = NULL;
	for (int b = 0; b < TEX_BANDS; b++)
	{
	    inject_SSTexId[0][b] = inject_SSTexId[1][b] = 0;
	}

	// initialize the multiple render target array
	for (int i = 0; i < 16; i++)
		mrts[i] = GL_COLOR_ATTACHMENT0 + i;

	vboID = 0;

	read_tex = 1; 
	write_tex = 0;

	inject_tex = 0;
	clean_tex = 1;

	cfactor = 1.0f;

	t_inc_camera_injection = NULL;
	t_inc_camera_cleanup = NULL;
	t_inc_light_injection = NULL;
	t_inc_light_cleanup = NULL;
}

GlobalIlluminationRendererIV::~GlobalIlluminationRendererIV()
{
	SAFEDELETE (injection_SS_shader);
	SAFEDELETE (injection_SS_Cleanup_shader);

	SAFEDELETE (propagation_shader);
	SAFEDELETE (shader);

	for (int i=0; i<6; i++)
		if (glIsTexture (propagateTexId[i]))
			glDeleteTextures (1, &(propagateTexId[i]));

	for (int i=0; i<2; i++)
		if (glIsFramebuffer (propagateFboId[i]))
			glDeleteFramebuffers (1, &(propagateFboId[i]));

	for (int b = 0; b < TEX_BANDS; b++)
	{
		if (glIsTexture (inject_SSTexId[0][b]))
			glDeleteTextures (1, &(inject_SSTexId[0][b]));
		if (glIsTexture (inject_SSTexId[1][b]))
			glDeleteTextures (1, &(inject_SSTexId[1][b]));
	}

	for (int i=0; i<2; i++)
		delete inject_SSFboId[i];

	if (glIsBuffer(vboID))
		glDeleteBuffers(1,&vboID);
	if (fbVerts!=NULL)
		delete (fbVerts);
	if (param_string)
		free (param_string);

#ifdef DO_TIMINGS
	delete t_inc_light_injection;
	delete t_inc_camera_injection;

	delete t_inc_light_cleanup;
	delete t_inc_camera_cleanup;
#endif
}

void GlobalIlluminationRendererIV::buildInjectionGrid()
{
	int width = inj_width;
	int height = inj_height;

	SAFEDELETE (fbVerts);
	fbVerts = new float[width*height*4];
	
	for (int j=0 ; j<height; j++){
		for (int i=0 ; i<width; i++){
			fbVerts[4*(j*width+i)+0]=float(i+0.5)/width;
			fbVerts[4*(j*width+i)+1]=float(j+0.5)/height;
			fbVerts[4*(j*width+i)+2]=0;	//depth will be fetched from zbuffer
			fbVerts[4*(j*width+i)+3]=1;
		}
	}
	
	if (glIsBuffer(vboID))
		glDeleteBuffers(1,&vboID);
	glGenBuffers(1,&vboID);
	glBindBuffer(GL_ARRAY_BUFFER, vboID);
	glBufferData(GL_ARRAY_BUFFER, 4*sizeof(float)*width*height, fbVerts, GL_STATIC_DRAW);
}

void GlobalIlluminationRendererIV::drawInjectionGrid()
{
	glBindBuffer(GL_ARRAY_BUFFER, vboID);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glDrawArrays(GL_POINTS, 0, inj_width*inj_height);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool GlobalIlluminationRendererIV::init(DeferredRenderer * _renderer)
{
	if (!initialized)
		if (!GlobalIlluminationRenderer::init(_renderer))
			return false;

#ifdef DO_TIMINGS
	t_inc_camera_injection = new Timer ("Incremental Camera Injection", 4);
	t_inc_camera_cleanup = new Timer ("Incremental Camera Cleanup", 3);

	t_inc_light_injection = new Timer ("Incremental Light Injection", 6);
	t_inc_light_cleanup = new Timer ("Incremental Light Cleanup", 5);
#endif

	// enable RSM rendering for GI lights
	set<int>::iterator ii = gi_lights.begin();
	for (ii=gi_lights.begin(); ii!=gi_lights.end(); ii++)  
	{
		DRLight * light = renderer->getLight(*ii);
		if (!light->isShadowEnabled())
			continue;
		light->enableExtendedData(true);
	}

	bool sinit = true;
	sinit &= injection_SS_shader->init(renderer);
	sinit &= injection_SS_Cleanup_shader->init(renderer);
	sinit &= propagation_shader->init(renderer);
	sinit &= shader->init(renderer);

	if (sinit)
	{
		initialized = true;
		return true;
	}
	else
		return false;
}

void GlobalIlluminationRendererIV::clearTexture3D()
{
	// clear the attached buffers
	for (int i = 0; i < 2; i++)
	{
		inject_SSFboId[i]->Bind ();
		glDrawBuffers (TEX_BANDS, mrts);

		for (unsigned int z = 0; z < m_depth; z++)
		{
			for (int b = 0; b < TEX_BANDS; b++)
				inject_SSFboId[i]->AttachTexture (GL_TEXTURE_3D, inject_SSTexId[i][b], GL_COLOR_ATTACHMENT0 + b, 0, z);
			inject_SSFboId[i]->IsValid ();

			// save the view port and set it to the size of the texture
			glPushAttrib (GL_VIEWPORT_BIT);
			glViewport (0, 0, m_width, m_height);

			glClearColor (0,0,0,0);
			glClear (GL_COLOR_BUFFER_BIT);

			// restore old view port
			glPopAttrib ();
		}
	}
}

bool GlobalIlluminationRendererIV::createTexture3D()
{
	bool sinit = true;

	resolution = renderer->getVolumeBufferResolution();
	m_bbox = renderer->getSceneRoot()->getBBox();

	Vector3D ratio;
	if (m_bbox.getMaxSide() == 0.0f)
		ratio = Vector3D(1.0f,1.0f,1.0f);
	else
	    ratio = m_bbox.getSize() / m_bbox.getMaxSide();

	// Create cube voxels
	m_width  = (int) floor (resolution * ratio.x + 0.5f);
	m_height = (int) floor (resolution * ratio.y + 0.5f);
	m_depth  = (int) floor (resolution * ratio.z + 0.5f);

	// Create cube volume	
//	m_width = m_height = m_depth = resolution;

	printf ("GlobalIlluminationRendererIV::createTexture3D():\n\tVoxelization grid requested: [%03d %03d %03d], actual: [%03d %03d %03d]\n\tVoxel size: [%.4f %.4f %.4f]\n",
		resolution, resolution, resolution,
		m_width, m_height, m_depth,
		m_bbox.getSize().x / m_width, m_bbox.getSize().y / m_height, m_bbox.getSize().z / m_depth);

	buildInjectionGrid();

	// generate the propagate textures
	for (int i=0; i<6; i++)
		if (glIsTexture (propagateTexId[i]))
			glDeleteTextures (1,&propagateTexId[i]);
	glGenTextures (6, propagateTexId);

	if (glIsFramebuffer (fbo))
		glDeleteFramebuffers (1, &fbo);
	glGenFramebuffers (1, &fbo);
	
	for (int i=0; i<2; i++)
		if (glIsFramebuffer (propagateFboId[i]))
			glDeleteFramebuffers (1, &(propagateFboId[i]));
	glGenFramebuffers (2, propagateFboId);

	for (int b = 0; b < TEX_BANDS; b++)
	{
		for (int i = 0; i < 2; i++)
		{
			if (glIsTexture (inject_SSTexId[i][b]))
				glDeleteTextures (1, &(inject_SSTexId[i][b]));
		}
	}

	for (int i = 0; i < 2; i++)
		delete inject_SSFboId[i];

	GLint mode = GL_CLAMP_TO_EDGE; // GL_CLAMP_TO_BORDER;
	GLint filter = GL_NEAREST;

	GLclampf priority = 1.0f;
	GLenum glErr = GL_NO_ERROR;

	for (int b = 0; b < TEX_BANDS && sinit; b++)
	{
		for (int i = 0; i < 2 && sinit; i++)
		{
			glGenTextures   (1, &inject_SSTexId[i][b]);
			glBindTexture   (GL_TEXTURE_3D, inject_SSTexId[i][b]);
			glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, filter);
			glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, filter);
			glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, mode);
			glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, mode);
			glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, mode);
			glTexParameteri (GL_TEXTURE_3D, GL_GENERATE_MIPMAP, GL_FALSE);
			glTexImage3D    (GL_TEXTURE_3D, 0, TEX_BITS, m_width, m_height, m_depth, 0, m_format, TEX_TYPE, 0);
		//	glPrioritizeTextures (1, &inject_SSTexId[i][b], &priority);
			if ((glErr = cwc::CheckGLErrorOutOfMemory ()) == GL_OUT_OF_MEMORY)
			{
				EAZD_TRACE ("GI Renderer (Incremental Volumes)::init() : ERROR - incomplete inject_SS texture objects. OUT OF MEMORY!");
				sinit = false;
			}
		}
	}

	if (! sinit) return false;

	for (int i = 0; i < 2; i++)
	{
		// create fbos
		inject_SSFboId[i] = new FramebufferObject ();

		// attach bank 0 of first texture to fbo for starters
		inject_SSFboId[i]->Bind ();
		inject_SSFboId[i]->AttachTexture (GL_TEXTURE_LAYER, inject_SSTexId[i][0], GL_COLOR_ATTACHMENT0 + 0, 0);
		sinit = inject_SSFboId[i]->IsValid ();
		inject_SSFboId[i]->Disable ();
	}
	CHECK_GL_ERROR ();

	if (! sinit) return false;

	// clear the attached buffers
	clearTexture3D();
	CHECK_GL_ERROR ();

	GLfloat borderColor[4] = {0.5, 0.5, 0.5, 0.5};

	//--------------- Propagation result buffers. 
	for(int i = 0; i<2; i++)
	{
		glBindFramebuffer (GL_FRAMEBUFFER, propagateFboId[i]);

		for(int j = 0; j<3; j++)
		{ 
			glBindTexture   (GL_TEXTURE_3D, propagateTexId[i*3+j]);
			glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, mode);
			glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, mode);
			glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, mode);
			glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, borderColor);
			glTexImage3D    (GL_TEXTURE_3D, 0, TEX_BITS, m_width, m_height, m_depth, 0, m_format, TEX_TYPE, 0);
			if ((glErr = cwc::CheckGLErrorOutOfMemory ()) == GL_OUT_OF_MEMORY)
			{
				EAZD_TRACE ("GI Renderer (Incremental Volumes)::init() : ERROR - incomplete propagate texture objects.");
				sinit = false;
			}
		}

		glFramebufferTexture3D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_3D, propagateTexId[i*3+0], 0, 0);
		glFramebufferTexture3D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_3D, propagateTexId[i*3+1], 0, 0);
		glFramebufferTexture3D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_3D, propagateTexId[i*3+2], 0, 0);

		if (glCheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		{
			EAZD_TRACE ("GI Renderer (Imperfect Volumes)::setupBuffer() : ERROR - incomplete frame buffer object (Propagator).");
			sinit = false;
		}
	}

	glBindFramebuffer (GL_FRAMEBUFFER, 0);    // Unbind the FBO for now

	if (sinit)
	{
		initialized = true;
		return true;
	}
	else
		return false;
}

unsigned int getPixelFormatNumComponents (unsigned int pixelFormat)
{
    switch (pixelFormat)
    {
        case GL_COLOR_INDEX: case GL_STENCIL_INDEX: case GL_DEPTH_COMPONENT: return 1;
        case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: case GL_ALPHA16F_ARB: case GL_ALPHA32F_ARB: return 1;
        case GL_LUMINANCE: case GL_LUMINANCE4: case GL_LUMINANCE8: case GL_LUMINANCE12: case GL_LUMINANCE16: case GL_LUMINANCE16F_ARB: case GL_LUMINANCE32F_ARB: return 1;
        case GL_INTENSITY: case GL_INTENSITY4: case GL_INTENSITY8: case GL_INTENSITY12: case GL_INTENSITY16: case GL_INTENSITY16F_ARB: case GL_INTENSITY32F_ARB: return 1;
		case GL_LUMINANCE32I_EXT: case GL_LUMINANCE32UI_EXT: case GL_ALPHA32I_EXT: case GL_ALPHA32UI_EXT: case GL_FLOAT_R32_NV: return 1;
        case GL_LUMINANCE4_ALPHA4: case GL_LUMINANCE6_ALPHA2: case GL_LUMINANCE8_ALPHA8: case GL_LUMINANCE12_ALPHA4: case GL_LUMINANCE12_ALPHA12: case GL_LUMINANCE16_ALPHA16: return 2;
        case GL_LUMINANCE_ALPHA: case GL_LUMINANCE_ALPHA16F_ARB: case GL_LUMINANCE_ALPHA32F_ARB: return 2;
        case GL_RGB: case GL_BGR: case GL_RGB16F: case GL_RGB32F: return 3;
		case GL_RGB32I: case GL_RGB32UI: case GL_FLOAT_RGB32_NV: return 3;
        case GL_RGBA: case GL_BGRA: case GL_RGBA16F: case GL_RGBA32F: return 4;
		case GL_FLOAT_RGBA32_NV: case GL_RGBA32I: case GL_RGBA32UI: case GL_RGBA8: return 4;

        default: return 0;
    }
}

unsigned int GlobalIlluminationRendererIV::getSizeData ()
{
	return getPixelFormatNumComponents (m_format) * m_width * m_height * m_depth;
}

float * GlobalIlluminationRendererIV::getData (GLint texId)
{
	unsigned int count = getSizeData ();

	float * data = (float *) malloc (count * sizeof (float));
//	memset (data, 0, count * sizeof (float));

	if (data == NULL)
	{
		EAZD_PRINT ("GlobalIlluminationRendererIV::getData() : INFO - Could not allocate the required CPU memory space to view the VBO.");
	}
	else
	{
		glBindTexture (GL_TEXTURE_3D, texId);
		glGetTexImage (GL_TEXTURE_3D, 0, m_format, GL_FLOAT, data);
		CHECK_GL_ERROR ();
	}

	return data;
}

// sides: 0: left, 1: top, 2: right, 3: bottom, 4: front, 5: back
void drawBoxSide (int i, GLfloat size, GLenum type)
{
	static GLfloat n[6][3] =
	{
		{-1.0,  0.0,  0.0},
		{ 0.0,  1.0,  0.0},
		{ 1.0,  0.0,  0.0},
		{ 0.0, -1.0,  0.0},
		{ 0.0,  0.0,  1.0},
		{ 0.0,  0.0, -1.0}
	};

	static GLint faces[6][4] =
	{
		{0, 1, 2, 3},
		{3, 2, 6, 7},
		{7, 6, 5, 4},
		{4, 5, 1, 0},
		{5, 6, 2, 1},
		{7, 4, 0, 3}
	};

	GLfloat v[8][3];

	v[0][0] = v[1][0] = v[2][0] = v[3][0] = -size * 0.5f;
	v[4][0] = v[5][0] = v[6][0] = v[7][0] =  size * 0.5f;
	v[0][1] = v[1][1] = v[4][1] = v[5][1] = -size * 0.5f;
	v[2][1] = v[3][1] = v[6][1] = v[7][1] =  size * 0.5f;
	v[0][2] = v[3][2] = v[4][2] = v[7][2] = -size * 0.5f;
	v[1][2] = v[2][2] = v[5][2] = v[6][2] =  size * 0.5f;

	glBegin (type);
	glNormal3fv (&n[i][0]);
	glVertex3fv (&v[faces[i][0]][0]);
	glVertex3fv (&v[faces[i][1]][0]);
	glVertex3fv (&v[faces[i][2]][0]);
	glVertex3fv (&v[faces[i][3]][0]);
	glEnd ();
} // end of drawBoxSide

void drawBox (GLfloat size)
{
	static GLfloat n[6][3] =
	{
		{-1.0,  0.0,  0.0},
		{ 0.0,  1.0,  0.0},
		{ 1.0,  0.0,  0.0},
		{ 0.0, -1.0,  0.0},
		{ 0.0,  0.0,  1.0},
		{ 0.0,  0.0, -1.0}
	};

	static GLint faces[6][4] =
	{
		{0, 1, 2, 3},
		{3, 2, 6, 7},
		{7, 6, 5, 4},
		{4, 5, 1, 0},
		{5, 6, 2, 1},
		{7, 4, 0, 3}
	};

	GLfloat v[8][3];

	v[0][0] = v[1][0] = v[2][0] = v[3][0] = -size * 0.5f;
	v[4][0] = v[5][0] = v[6][0] = v[7][0] =  size * 0.5f;
	v[0][1] = v[1][1] = v[4][1] = v[5][1] = -size * 0.5f;
	v[2][1] = v[3][1] = v[6][1] = v[7][1] =  size * 0.5f;
	v[0][2] = v[3][2] = v[4][2] = v[7][2] = -size * 0.5f;
	v[1][2] = v[2][2] = v[5][2] = v[6][2] =  size * 0.5f;

	for (int i = 5; i >= 0; i--)
	{
		glBegin (GL_QUADS);
		glNormal3fv (&n[i][0]);
		glVertex3fv (&v[faces[i][0]][0]);
		glVertex3fv (&v[faces[i][1]][0]);
		glVertex3fv (&v[faces[i][2]][0]);
		glVertex3fv (&v[faces[i][3]][0]);
		glEnd ();
	}
} // end of drawBox


void drawWireBox (GLfloat size)
{
	static GLfloat n[6][3] =
	{
		{-1.0,  0.0,  0.0},
		{ 0.0,  1.0,  0.0},
		{ 1.0,  0.0,  0.0},
		{ 0.0, -1.0,  0.0},
		{ 0.0,  0.0,  1.0},
		{ 0.0,  0.0, -1.0}
	};

	static GLint faces[6][4] =
	{
		{0, 1, 2, 3},
		{3, 2, 6, 7},
		{7, 6, 5, 4},
		{4, 5, 1, 0},
		{5, 6, 2, 1},
		{7, 4, 0, 3}
	};

	GLfloat v[8][3];

	v[0][0] = v[1][0] = v[2][0] = v[3][0] = -size * 0.5f;
	v[4][0] = v[5][0] = v[6][0] = v[7][0] =  size * 0.5f;
	v[0][1] = v[1][1] = v[4][1] = v[5][1] = -size * 0.5f;
	v[2][1] = v[3][1] = v[6][1] = v[7][1] =  size * 0.5f;
	v[0][2] = v[3][2] = v[4][2] = v[7][2] = -size * 0.5f;
	v[1][2] = v[2][2] = v[5][2] = v[6][2] =  size * 0.5f;

	glColor3f (1.0, 0.5, 0.0);

	for (int i = 5; i >= 0; i--)
	{
		glBegin (GL_LINE_LOOP);
		glNormal3fv (&n[i][0]);
		glVertex3fv (&v[faces[i][0]][0]);
		glVertex3fv (&v[faces[i][1]][0]);
		glVertex3fv (&v[faces[i][2]][0]);
		glVertex3fv (&v[faces[i][3]][0]);
		glEnd ();
	}
} // end of drawWireBox

void GlobalIlluminationRendererIV::drawTexture3D (GLint texId)
{
	// read back the pixels
	float * buffer = getData (texId);
	if (buffer == NULL) return;

	Vector4D clear_color = Vector4D(0,0,0,0);
	BBox3D bbox = m_bbox;

	Vector3D min, max;
    min.x = bbox.getMin().x - bbox.getSize ().x * (0.5f / ((int) resolution));
    min.y = bbox.getMin().y - bbox.getSize ().y * (0.5f / ((int) resolution));
    min.z = bbox.getMin().z - bbox.getSize ().z * (0.5f / ((int) resolution));
    max.x = bbox.getMax().x + bbox.getSize ().x * (0.5f / ((int) resolution));
    max.y = bbox.getMax().y + bbox.getSize ().y * (0.5f / ((int) resolution));
    max.z = bbox.getMax().z + bbox.getSize ().z * (0.5f / ((int) resolution));
	bbox.set (min, max);

	Vector3D thickness = Vector3D (
		(bbox.getSize ().x) / m_width,
		(bbox.getSize ().y) / m_height,
		(bbox.getSize ().z) / m_depth);

	glPolygonMode(GL_FRONT, GL_FILL);

	glPushAttrib (GL_ENABLE_BIT);
	glEnable (GL_CULL_FACE);
	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// get the number of components
	unsigned int numComponents = getPixelFormatNumComponents (m_format);
	unsigned int i, j, k;

	// process the read pixels
	for (k = 0; k < m_depth; k++)
	for (j = 0; j < m_height; j++)
	for (i = 0; i < m_width; i++)
	{
		unsigned int index = ((k * m_height + j) * m_width + i) * numComponents; // access a 3d texture with a single index

		if (! (buffer[index+0] == clear_color.x
			&& buffer[index+1] == clear_color.y
			&& buffer[index+2] == clear_color.z
			&& buffer[index+3] == clear_color.w
			))
		{
			glColor3f (MAP_MINUS1TO1_0TO1 (buffer[index+0]), MAP_MINUS1TO1_0TO1 (buffer[index+1]), MAP_MINUS1TO1_0TO1 (buffer[index+2]));

			glMatrixMode(GL_MODELVIEW);
			glPushMatrix ();
				glTranslatef (bbox.getMin().x + (i+0.5f) * thickness.x,
							  bbox.getMin().y + (j+0.5f) * thickness.y,
							  bbox.getMin().z + (k+0.5f) * thickness.z);
				glScalef (thickness.x, thickness.y, thickness.z);
				drawBox (1.0f);
				drawWireBox (1.0f);
			glPopMatrix ();
		}
	}

	glPopAttrib ();

	free (buffer);
} // end of drawTexture3D

void GlobalIlluminationRendererIV::draw()
{
	if (!renderer)
		return;

	static bool first = true;
	if (first)
	{
		first = false;
		if (! createTexture3D ())
			return;
	}
	CHECK_GL_ERROR ();

	World3D *root = renderer->getSceneRoot();

	glPushAttrib(GL_VIEWPORT_BIT);
	glDisable(GL_ALPHA_TEST);
	
	// decode params string at run time to receive the most recent value;
	if (!params_decoded && param_string)
	{
		// get params here
		params_decoded = true;
		sscanf (param_string, "%f", &cfactor);
	}

	// re-adjust the camera-space reprojection grid size, if necessary
	inj_grid_size = 2 * resolution;

	if (inj_width!=inj_grid_size
	|| inj_height!=inj_grid_size)
	{
		inj_width=inj_grid_size;
		inj_height=inj_grid_size;
	    buildInjectionGrid();
	}

	// get BBOX and store the WCS->3DTex transform
	m_bbox = renderer->getBoundingBox();

	Vector3D mn,mx;
	mn = m_bbox.getMin();
	mx = m_bbox.getMax();

	Matrix4D Ms = Matrix4D();
	Matrix4D Mt = Matrix4D();
	Vector3D sfactor = Vector3D(1.0f/m_bbox.getSize().x, 1.0f/m_bbox.getSize().y, 1.0f/m_bbox.getSize().z);
	Ms.makeScale(sfactor);
	Mt.makeTranslate(-m_bbox.getMin()+sfactor*0.5f);
	Matrix4D final = Ms * Mt;
	final.transpose();
	for (int i=0; i<16; i++)
		transform[i]=final[i];

	glBlendEquation (GL_ADD);
    glBlendFunc (GL_ONE,GL_ZERO);
	glDisable (GL_BLEND);
	glDisable (GL_ALPHA);
	glDisable (GL_DEPTH);

#ifdef DO_TIMINGS
//	t_injection->start();
#endif

	// fill in every slice of the photon map

	glViewport(0,0,m_width,m_height);
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(mn.x,mx.x,mn.y,mx.y, 0.001, -(mx.z-mn.z)-0.001);
	glTranslatef(0,0,-mn.z);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


//-------------------------------------------------------------- INCREMENTAL -----------------------------------------------	
//--------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------- BEGIN --------------------------------------------------

//------------------------------------------------------------- CLEANUP STAGE ----------------------------------------------
	
	// When enabled this effectively simulates single-frame non-incremental voxelization
//	clearTexture3D();

	set<int>::iterator jj = gi_lights.begin();

if (inj_lights)
{
	// iterate for every GI light
	for (jj=gi_lights.begin(); jj!=gi_lights.end(); jj++)
	{
		int lid = *jj;
		DRLight * light = renderer->getLight(lid);
		if (! light->isActive())
			continue;

		GLdouble * lmat, lmat_inv[16];
		lmat = renderer->getLightMatrix(lid);
		invert_matrix(lmat, lmat_inv);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(-1,1,-1,1,-1,1);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glMatrixMode(GL_TEXTURE_MATRIX);
		glActiveTexture(GL_TEXTURE0);
		glLoadIdentity();
		glActiveTexture(GL_TEXTURE1);
		glLoadIdentity();
	
		inject_SSFboId[clean_tex]->Bind ();
		glDrawBuffers (TEX_BANDS, mrts);

		glActiveTexture(GL_TEXTURE0); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, light->getShadowMap());
		glActiveTexture(GL_TEXTURE1); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, light->getShadowMapColorBuffer());
		glActiveTexture(GL_TEXTURE2); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, light->getShadowMapNormalBuffer());

		// bind 3D texture for reading
		for (int b = 0; b < TEX_BANDS; b++)
		{
			glActiveTexture(GL_TEXTURE4 + b); glEnable(GL_TEXTURE_3D); glBindTexture(GL_TEXTURE_3D, inject_SSTexId[inject_tex][b]);
		}

#ifdef DO_TIMINGS
	t_inc_light_cleanup->start();
#endif
	
		// slice the z-direction
		for (unsigned int z = 0; z < m_depth; z++)
		{
			// attach texture slice to FBO for writing
			for (int b = 0; b < TEX_BANDS; b++)
				inject_SSFboId[clean_tex]->AttachTexture (GL_TEXTURE_3D, inject_SSTexId[clean_tex][b], GL_COLOR_ATTACHMENT0 + b, 0, z);
			inject_SSFboId[clean_tex]->IsValid ();

			injection_SS_Cleanup_shader->setVolWidth((float) m_width);
			injection_SS_Cleanup_shader->setVolHeight((float) m_height);
			injection_SS_Cleanup_shader->setVoxelRadius((m_bbox.getSize() / Vector3D ((float) m_width, (float) m_height, (float) m_depth)).length () * 0.5f);
			injection_SS_Cleanup_shader->setVoxelHalfSize(m_bbox.getSize() / Vector3D ((float) m_width, (float) m_height, (float) m_depth) * 0.5f);
			injection_SS_Cleanup_shader->setPrebakedLighting(true);
			injection_SS_Cleanup_shader->setModelViewProjectionMatrix(lmat);
			injection_SS_Cleanup_shader->setModelViewProjectionMatrixInverse(lmat_inv);
			float lmatPinv[16];
			double lmatPinvd[16];
			for (int k=0;k<16;k++)
				 lmatPinv[k] = (float)light->getProjectionMatrix()[k];
			Matrix4D LPinv = Matrix4D(lmatPinv);
			LPinv.invert();
			for (int k=0;k<16;k++)
				 lmatPinvd[k] = LPinv.a[k];
			injection_SS_Cleanup_shader->setProjectionMatrixInverse(lmatPinvd);
			injection_SS_Cleanup_shader->setProjectionMatrix(light->getProjectionMatrix());
			injection_SS_Cleanup_shader->setCOP(light->getTransformedPosition());
			glDisable(GL_BLEND);
		
			injection_SS_Cleanup_shader->start();
	
			glBegin (GL_QUADS);
				glNormal3f (0.0f, 0.0f, 1.0f);
				float gz = mn.z+(mx.z-mn.z)*(z + 0.5f) / (float) m_depth;
				glMultiTexCoord3fARB( GL_TEXTURE0_ARB, mn.x, mn.y, gz );
				glVertex3f(-1,-1,-1+2*(z + 0.5f) / (float) m_depth);
			
				glMultiTexCoord3fARB( GL_TEXTURE0_ARB, mx.x, mn.y, gz);
				glVertex3f(1,-1,-1+2*(z + 0.5f) / (float) m_depth);
			
				glMultiTexCoord3fARB( GL_TEXTURE0_ARB, mx.x, mx.y, gz);
				glVertex3f(1,1,-1+2*(z + 0.5f) / (float) m_depth);
			
				glMultiTexCoord3fARB( GL_TEXTURE0_ARB, mn.x, mx.y, gz);
				glVertex3f(-1,1,-1+2*(z + 0.5f) / (float) m_depth);
			glEnd();

			injection_SS_Cleanup_shader->stop();
	
		} // for all slices

#ifdef DO_TIMINGS
	t_inc_light_cleanup->stop();
#endif

		// disable here all texture units used
		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D);
		glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D);
		glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D);

		for (int b = 0; b < TEX_BANDS; b++)
		{
			glActiveTexture(GL_TEXTURE4 + b); glBindTexture(GL_TEXTURE_3D, 0); glDisable(GL_TEXTURE_3D);
		}

		// swap read and write indexes
		if (inject_tex) { inject_tex = 0; clean_tex = 1; }
		else			{ inject_tex = 1; clean_tex = 0; }
	
	} // for all lights
}

if (inj_camera)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1,1,-1,1,-1,1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_TEXTURE_MATRIX);
	glLoadIdentity();
	
	inject_SSFboId[clean_tex]->Bind ();
    glDrawBuffers (TEX_BANDS, mrts);

	glActiveTexture(GL_TEXTURE0); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, renderer->getBuffer(DR_TARGET_DEPTH));
	glActiveTexture(GL_TEXTURE1); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, renderer->getBuffer(DR_TARGET_COLOR));
	glActiveTexture(GL_TEXTURE2); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, renderer->getBuffer(DR_TARGET_NORMAL));
	glActiveTexture(GL_TEXTURE3); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, renderer->getBuffer(DR_TARGET_LIGHTING));
	
	glDisable(GL_ALPHA_TEST);
	glDisable(GL_DEPTH_TEST);
	glBlendEquation(GL_FUNC_ADD);
	glBlendFunc(GL_ONE,GL_ZERO);

#ifdef DO_TIMINGS
	t_inc_camera_cleanup->start();
#endif

	// slice the z-direction
	for (unsigned int z = 0; z < m_depth; z++)
	{
		// attach texture slice to FBO for writing
		for (int b = 0; b < TEX_BANDS; b++)
			inject_SSFboId[clean_tex]->AttachTexture (GL_TEXTURE_3D, inject_SSTexId[clean_tex][b], GL_COLOR_ATTACHMENT0 + b, 0, z);
		inject_SSFboId[clean_tex]->IsValid ();

		// bind 3D texture for reading
		for (int b = 0; b < TEX_BANDS; b++)
		{
			glActiveTexture(GL_TEXTURE4 + b); glEnable(GL_TEXTURE_3D); glBindTexture(GL_TEXTURE_3D, inject_SSTexId[inject_tex][b]);
		}

		injection_SS_Cleanup_shader->setVolWidth((float) m_width);
		injection_SS_Cleanup_shader->setVolHeight((float) m_height);
		injection_SS_Cleanup_shader->setVoxelRadius((m_bbox.getSize() / Vector3D ((float) m_width, (float) m_height, (float) m_depth)).length () * 0.5f);
		injection_SS_Cleanup_shader->setVoxelHalfSize(m_bbox.getSize() / Vector3D ((float) m_width, (float) m_height, (float) m_depth) * 0.5f);
		injection_SS_Cleanup_shader->setPrebakedLighting(false);
		injection_SS_Cleanup_shader->setModelViewProjectionMatrix(renderer->getModelViewProjectionMatrix());
		injection_SS_Cleanup_shader->setModelViewProjectionMatrixInverse(renderer->getModelViewProjectionMatrixInv());
		
		injection_SS_Cleanup_shader->setProjectionMatrixInverse(renderer->getProjectionMatrixInv());
		injection_SS_Cleanup_shader->setProjectionMatrix(renderer->getProjectionMatrix());
		injection_SS_Cleanup_shader->setCOP(root->getActiveCamera()->getWorldPosition());

		injection_SS_Cleanup_shader->start();
	
		glDisable(GL_BLEND);
		glBegin (GL_QUADS);
			glNormal3f (0.0f, 0.0f, 1.0f);
			float gz = mn.z+(mx.z-mn.z)*(z + 0.5f) / (float) m_depth;
			glMultiTexCoord3fARB( GL_TEXTURE0_ARB, mn.x, mn.y, gz );
			glVertex3f(-1,-1,-1+2*(z + 0.5f) / (float) m_depth);
			
			glMultiTexCoord3fARB( GL_TEXTURE0_ARB, mx.x, mn.y, gz);
			glVertex3f(1,-1,-1+2*(z + 0.5f) / (float) m_depth);
			
			glMultiTexCoord3fARB( GL_TEXTURE0_ARB, mx.x, mx.y, gz);
			glVertex3f(1,1,-1+2*(z + 0.5f) / (float) m_depth);
			
			glMultiTexCoord3fARB( GL_TEXTURE0_ARB, mn.x, mx.y, gz);
			glVertex3f(-1,1,-1+2*(z + 0.5f) / (float) m_depth);
		glEnd();

		injection_SS_Cleanup_shader->stop();

		// disable here all texture units used
		for (int b = 0; b < TEX_BANDS; b++)
		{
			glActiveTexture(GL_TEXTURE4 + b); glBindTexture(GL_TEXTURE_3D, 0); glDisable(GL_TEXTURE_3D);
		}

		//	inject_SSFboId[clean_tex]->Disable ();
	} // for all slices

#ifdef DO_TIMINGS
	t_inc_camera_cleanup->stop();
#endif

	// disable here all texture units used
	glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D);

	// swap read and write indexes
	if (inject_tex) { inject_tex = 0; clean_tex = 1; }
	else			{ inject_tex = 1; clean_tex = 0; }
}

//----------------------------------------------------------- INJECTION STAGE ----------------------------------------------

	inject_SSFboId[inject_tex]->Bind ();
	glDrawBuffers (TEX_BANDS, mrts); 

	// attach texture slice to FBO for writing
	for (int b = 0; b < TEX_BANDS; b++)
		inject_SSFboId[inject_tex]->AttachTexture (GL_TEXTURE_LAYER, inject_SSTexId[inject_tex][b], GL_COLOR_ATTACHMENT0 + b, 0);
	inject_SSFboId[inject_tex]->IsValid ();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(mn.x,mx.x,mn.y,mx.y, 0.0001, -(mx.z-mn.z)-0.0001);
	glTranslatef(0,0,-mn.z);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_TEXTURE_MATRIX);
	glActiveTexture(GL_TEXTURE0);
	glLoadIdentity();
	glActiveTexture(GL_TEXTURE1);
	glLoadIdentity();

if (inj_lights)
{
	// iterate for every GI light
	
	for (jj=gi_lights.begin(); jj!=gi_lights.end(); jj++)
	{
		int lid = *jj;
		DRLight * light = renderer->getLight(lid);
		if (! light->isActive())
			continue;

		GLdouble * lmat, lmat_inv[16];
		lmat = renderer->getLightMatrix(lid);
		invert_matrix(lmat, lmat_inv);
		glActiveTexture(GL_TEXTURE0); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, light->getShadowMap());
		glActiveTexture(GL_TEXTURE1); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, light->getShadowMapColorBuffer());
		glActiveTexture(GL_TEXTURE2); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, light->getShadowMapNormalBuffer());

		injection_SS_shader->setVolDepth(m_depth);
		injection_SS_shader->setPrebakedLighting(true);
		injection_SS_shader->setModelViewProjectionMatrixInverse(lmat_inv);
		injection_SS_shader->setProjectionMatrix(light->getProjectionMatrix());

#ifdef DO_TIMINGS
	t_inc_light_injection->start();
#endif

		injection_SS_shader->start();
		drawInjectionGrid();
		injection_SS_shader->stop();

#ifdef DO_TIMINGS
	t_inc_light_injection->stop();
#endif

		// disable here all texture units used
		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D);
		glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D);
		glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D);

	} // for every light
}

if (inj_camera)
{
//	glDisable(GL_TEXTURE_3D);

	glActiveTexture(GL_TEXTURE0); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, renderer->getBuffer(DR_TARGET_DEPTH));
	glActiveTexture(GL_TEXTURE1); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, renderer->getBuffer(DR_TARGET_COLOR));
	glActiveTexture(GL_TEXTURE2); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, renderer->getBuffer(DR_TARGET_NORMAL));
	glActiveTexture(GL_TEXTURE3); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, renderer->getBuffer(DR_TARGET_LIGHTING));

	injection_SS_shader->setVolDepth(m_depth);
	injection_SS_shader->setPrebakedLighting(false);
	injection_SS_shader->setModelViewProjectionMatrixInverse(renderer->getModelViewProjectionMatrixInv());
	injection_SS_shader->setProjectionMatrix(renderer->getProjectionMatrix());

#ifdef DO_TIMINGS
	t_inc_camera_injection->start();
#endif

	injection_SS_shader->start();
	drawInjectionGrid();
	injection_SS_shader->stop();

#ifdef DO_TIMINGS
	t_inc_camera_injection->stop();
#endif

	// disable here all texture units used
	glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D);

//	inject_SSFboId[inject_tex]->Disable ();
}

	FramebufferObject::Disable ();

//-------------------------------------------------------------- INCREMENTAL -----------------------------------------------	
//--------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------ END ---------------------------------------------------

    glBlendEquation (GL_FUNC_ADD);
	glBlendFunc (GL_ONE,GL_ZERO);

//-------------------------------------------------------------- PROPAGATION -----------------------------------------------	
//--------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------- BEGIN --------------------------------------------------

	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glBlendEquation (GL_FUNC_ADD);
	glBlendFunc(GL_ONE,GL_ONE);
	
	int p;
	for (p = 0; p < samples; p++)
	{
		glBindFramebuffer (GL_FRAMEBUFFER, propagateFboId[write_tex]);

																							// the first time read from the composited texture
		glActiveTexture(GL_TEXTURE10); glEnable(GL_TEXTURE_3D); glBindTexture(GL_TEXTURE_3D, p == 0 ? inject_SSTexId[inject_tex][1] : propagateTexId[read_tex*3+0]);
		glActiveTexture(GL_TEXTURE11); glEnable(GL_TEXTURE_3D); glBindTexture(GL_TEXTURE_3D, p == 0 ? inject_SSTexId[inject_tex][2] : propagateTexId[read_tex*3+1]);
		glActiveTexture(GL_TEXTURE12); glEnable(GL_TEXTURE_3D); glBindTexture(GL_TEXTURE_3D, p == 0 ? inject_SSTexId[inject_tex][3] : propagateTexId[read_tex*3+2]);
		glActiveTexture(GL_TEXTURE13); glEnable(GL_TEXTURE_3D); glBindTexture(GL_TEXTURE_3D, inject_SSTexId[inject_tex][0]);

		const GLenum  _mrts[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };

		for (int z=0; z<m_depth; z++)
		{
			glFramebufferTexture3D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_3D, propagateTexId[3*write_tex+0], 0, z);
			glFramebufferTexture3D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_3D, propagateTexId[3*write_tex+1], 0, z);
			glFramebufferTexture3D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_3D, propagateTexId[3*write_tex+2], 0, z);
			glDrawBuffers (3, _mrts); 

			if (glCheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			{
				EAZD_TRACE ("GI Renderer : ERROR - incomplete frame buffer object (Propagation).");
			}

			glViewport(0,0,m_width,m_height);
			glClearColor(0,0,0,0.0);
			glClear(GL_COLOR_BUFFER_BIT);
			
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(0, 1, 0, 1, -z/(float)m_depth, -(z+1)/(float)m_depth);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			
			propagation_shader->setIteration(p);
			propagation_shader->setDimensions(m_width, m_height, m_depth);
			propagation_shader->setSpreadFactor(cfactor);
			
			propagation_shader->start();

			glBegin(GL_QUADS);
			glNormal3f(0,0,1);
			glVertex3f(0,0,(z+0.5f)/(float)m_depth);
			glVertex3f(1,0,(z+0.5f)/(float)m_depth);
			glVertex3f(1,1,(z+0.5f)/(float)m_depth);
			glVertex3f(0,1,(z+0.5f)/(float)m_depth);
			glEnd();

			propagation_shader->stop();
		}

		// swap read and write indexes
		if (read_tex) { read_tex = 0; write_tex = 1; }
		else		  { read_tex = 1; write_tex = 0; }
	} // for samples
	
	CHECK_GL_ERROR ();

	if (p == 0)             // no propagation
	{
		photonMapTexId[0] = inject_SSTexId[clean_tex][0];
		photonMapTexId[1] = inject_SSTexId[clean_tex][1];
		photonMapTexId[2] = inject_SSTexId[clean_tex][2];
	}
	else
	{
		photonMapTexId[0] = propagateTexId[0];
		photonMapTexId[1] = propagateTexId[1];
		photonMapTexId[2] = propagateTexId[2];
	}

	CHECK_GL_ERROR ();

//-------------------------------------------------------------- PROPAGATION -----------------------------------------------	
//--------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------ END ---------------------------------------------------

	glPopAttrib();
	glDrawBuffer(0);

//---------------------------------------------------------------- DRAW GI -------------------------------------------------	
//--------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------- BEGIN --------------------------------------------------


	// ---------------------------- Render GI to the GI (AO) buffer
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, renderer->getAOfbo());
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)!=GL_FRAMEBUFFER_COMPLETE_EXT)
	{
		EAZD_TRACE ("GI Renderer (IV) : ERROR - incomplete frame buffer object (GI pass).");
	}

	glBlendEquation(GL_FUNC_ADD);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	
	shader->setModelViewProjectionMatrix(renderer->getModelViewProjectionMatrix());
	shader->setModelViewProjectionMatrixInverse(renderer->getModelViewProjectionMatrixInv());
	shader->setProjectionMatrix(renderer->getProjectionMatrix());
	shader->setProjectionMatrixInverse(renderer->getProjectionMatrixInv());
	((DRShaderGI_IV*)shader)->setBoundingBox(renderer->getBoundingBox());
	((DRShaderGI_IV*)shader)->setDimensions(m_width, m_height, m_depth);

	
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_TEXTURE_3D);
	glMatrixMode(GL_TEXTURE);
	glActiveTextureARB(GL_TEXTURE0);
	glLoadIdentity();
    glActiveTextureARB(GL_TEXTURE1);
    glLoadMatrixf(transform);

    glActiveTextureARB(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, renderer->getBuffer(DR_TARGET_NORMAL));
	glActiveTexture   (GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, renderer->getBuffer(DR_TARGET_DEPTH));
	glActiveTextureARB(GL_TEXTURE7); glBindTexture(GL_TEXTURE_3D, photonMapTexId[0]);
	glActiveTextureARB(GL_TEXTURE8); glBindTexture(GL_TEXTURE_3D, photonMapTexId[1]);
	glActiveTextureARB(GL_TEXTURE9); glBindTexture(GL_TEXTURE_3D, photonMapTexId[2]);
    
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1,1,-1,1,1,-1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	shader->start();
		
	glBegin(GL_QUADS);
		glColor4f(1,1,1,1);
		glNormal3f(0,0,1);
		glTexCoord2f(0,1);	glVertex3f(-1,1,0);
		glTexCoord2f(0,0);	glVertex3f(-1,-1,0);
		glTexCoord2f(1,0);	glVertex3f(1,-1,0);
		glTexCoord2f(1,1);	glVertex3f(1,1,0);
	glEnd();

	shader->stop();

	CHECK_GL_ERROR ();

//---------------------------------------------------------------- DRAW GI -------------------------------------------------	
//--------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------ END ---------------------------------------------------

	glEnable(GL_BLEND);

	glDrawBuffer(0);

	for (int i=0; i<16; i++)
	{
		glActiveTextureARB(GL_TEXTURE0+i);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	glColorMask(1,1,1,1);
}
