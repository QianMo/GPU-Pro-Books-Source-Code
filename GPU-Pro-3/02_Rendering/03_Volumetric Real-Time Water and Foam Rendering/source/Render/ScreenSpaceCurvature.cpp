#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include "GL/glew.h"
#include <GL/gl.h>

#include <stdio.h>
#include <assert.h>

#include <vector>
#include <limits>

#include "../Render/ScreenSpaceCurvature.h"
#include "../Render/ShaderManager.h"

#include "../Physic/Fluid.h"

#include "../Util/Vector3.h"
#include "../Util/perlin.h"

#include <GL/glut.h>

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

#define NOISE_WIDTH 32
#define NOISE_HEIGHT 32
#define NOISE_DEPTH 32

#define FOAM_NOISE_WIDTH 32
#define FOAM_NOISE_HEIGHT 32
#define FOAM_NOISE_DEPTH 32

#define BYTES_PER_TEXEL 3

#define LAYER(r)	(NOISE_WIDTH * NOISE_HEIGHT * r * BYTES_PER_TEXEL)
#define TEXEL2(s, t)	(BYTES_PER_TEXEL * (s * NOISE_WIDTH + t))
#define TEXEL3(s, t, r)	(TEXEL2(s, t) + LAYER(r))

#define FOAM_LAYER(r)	(FOAM_NOISE_WIDTH * FOAM_NOISE_HEIGHT * r * BYTES_PER_TEXEL)
#define FOAM_TEXEL2(s, t)	(BYTES_PER_TEXEL * (s * FOAM_NOISE_WIDTH + t))
#define FOAM_TEXEL3(s, t, r)	(FOAM_TEXEL2(s, t) + FOAM_LAYER(r))

float ScreenSpaceCurvature::EPSILON = 0.05f;
float ScreenSpaceCurvature::THRESHOLD_MIN = -200.0f;
float ScreenSpaceCurvature::BLUR_DEPTH_FALLOFF = 5.0f;
float ScreenSpaceCurvature::DEPTH_THRESHOLD = 5.0f;


// -----------------------------------------------------------------------------
// ----------------- ScreenSpaceCurvature::ScreenSpaceCurvature ----------------
// -----------------------------------------------------------------------------
ScreenSpaceCurvature::ScreenSpaceCurvature(void) :
	fluid(NULL),
	fluidMetaData(NULL),
	bufferSize(0.0f, 0.0f),
	frameBuffer(0),
	renderBuffer(0),
	depthRenderBuffer(0),
	windowWidth(0),
	windowHeight(0),
	scaleDownWidth(0),
	scaleDownHeight(0),
	lowResFactor(1.0f),
	fieldOfView(0.0f),
	invFocalLength(0.0f),
	aspectRatio(0.0f),
	sceneTexture(0),
	foamDepthTexture(0),
	depthTexture(0),
	thicknessTexture(0),
	foamThicknessTexture(0),
	smoothedDepthTexture(0),
	noiseTexture(0),
	resultTexture(0),
	perlinNoiseTexture(0),
	foamPerlinNoiseTexture(0),
	currentDepthSource(0),
	fluidLightPosition(Vector3::ZERO),
	vbo(0),
	visiblePacketCount(0),
	vboStartIndices(NULL),
	vboIndexCount(NULL),

	occQuery(0),
	currentIterationCount(0)
{
	unsigned int i;

#if USE_DOWNSAMPLE_SHIFT
	for (i=0; i<DOWNSAMPLE_SHIFT; i++)
	{
		downsampleDepthTexture[i] = 0;
		downsampleFoamDepthTexture[i] = 0;
	}
#endif
}


// -----------------------------------------------------------------------------
// ------------- ScreenSpaceCurvature::~ScreenSpaceCurvature -------------------
// -----------------------------------------------------------------------------
ScreenSpaceCurvature::~ScreenSpaceCurvature(void)
{
	Exit();
}


// -----------------------------------------------------------------------------
// --------------------- ScreenSpaceCurvature::Init ----------------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::Init(void)
{
#if USE_DOWNSAMPLE_SHIFT
	lowResFactor = 1.0f/(Math::Pow(2, DOWNSAMPLE_SHIFT));
#endif

	bufferSize.x = (float)windowWidth;
	bufferSize.y = (float)windowHeight;

	// FBO
	glGenFramebuffersEXT(1, &frameBuffer);

	glGenRenderbuffersEXT(1, &renderBuffer);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderBuffer);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT32_ARB, windowWidth, windowHeight);
	CheckFrameBufferState();

	glGenRenderbuffersEXT(1, &depthRenderBuffer);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthRenderBuffer);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT32_ARB, windowWidth, windowHeight);
	CheckFrameBufferState();

	// textures
	glEnable(GL_TEXTURE_RECTANGLE_ARB);

	GLenum target = GL_TEXTURE_RECTANGLE_ARB;

	sceneTexture = CreateTexture(target, windowWidth, windowHeight, GL_RGBA8, GL_RGBA);

	thicknessTexture = CreateTexture(target, windowWidth, windowHeight, GL_RGB16F_ARB, GL_LUMINANCE);
	foamThicknessTexture = CreateTexture(target, windowWidth, windowHeight, GL_RGB16F_ARB, GL_LUMINANCE);

#if USE_DOWNSAMPLE_SHIFT
	depthTexture = CreateTexture(target, windowWidth, windowHeight, GL_RGB32F_ARB, GL_LUMINANCE);
	foamDepthTexture = CreateTexture(target, windowWidth, windowHeight, GL_LUMINANCE16F_ARB, GL_LUMINANCE);

	for(int i=0; i<DOWNSAMPLE_SHIFT; ++i)
	{
		downsampleDepthTexture[i] = CreateTexture(target, windowWidth >> (i+1), windowHeight >> (i+1), GL_RGB32F_ARB, GL_LUMINANCE);
		downsampleFoamDepthTexture[i] = CreateTexture(target, windowWidth >> (i+1), windowHeight >> (i+1), GL_LUMINANCE16F_ARB, GL_LUMINANCE);
	}

	smoothedDepthTexture = CreateTexture(target, windowWidth >> DOWNSAMPLE_SHIFT, windowHeight >> DOWNSAMPLE_SHIFT, GL_RGB32F_ARB, GL_LUMINANCE);
#else
	depthTexture = CreateTexture(target, windowWidth, windowHeight, GL_LUMINANCE32F_ARB, GL_LUMINANCE);
	foamDepthTexture = CreateTexture(target, windowWidth, windowHeight, GL_LUMINANCE16F_ARB, GL_LUMINANCE);

	smoothedDepthTexture = CreateTexture(target, windowWidth, windowHeight, GL_LUMINANCE32F_ARB, GL_LUMINANCE);
#endif

	noiseTexture = CreateTexture(target, windowWidth, windowHeight, GL_LUMINANCE16F_ARB, GL_LUMINANCE);

	resultTexture = CreateTexture(target, windowWidth, windowHeight, GL_RGBA8, GL_RGBA);

	BuildNoiseTexture();
	BuildFoamNoiseTexture();

	// VBO
	int bufferSize;
	if (vbo > 0)
	{
		glDeleteBuffersARB(1, &vbo);
		vbo = 0;
	}

	glGenBuffersARB(1, &vbo);	
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, fluid->GetMaxParticles()*sizeof(Fluid::FluidParticle), NULL, GL_STREAM_DRAW_ARB);
	glGetBufferParameterivARB(GL_ARRAY_BUFFER_ARB, GL_BUFFER_SIZE_ARB, &bufferSize);

	vboStartIndices = new int[fluid->GetMaxPackets()];
	vboIndexCount = new int[fluid->GetMaxPackets()];

	// fluid stuff
	fluidMetaData = new MetaData[fluid->GetMaxParticles()];
	memset(fluidMetaData, 0, sizeof(MetaData)*fluid->GetMaxParticles());

	// occlusion query
	glGenQueriesARB(1, &occQuery);
}


// -----------------------------------------------------------------------------
// -------------------- ScreenSpaceCurvature::Update ---------------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::Update(float deltaTime)
{
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_RENDER_MODE, (float)renderMode);

	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_USE_NOISE, renderDescription.useNoise);

	ShaderManager::Instance()->SetParameter4fv(ShaderManager::SP_FLUID_BASE_COLOR, renderDescription.baseColor.c);
	ShaderManager::Instance()->SetParameter4fv(ShaderManager::SP_FLUID_COLOR_FALLOFF, renderDescription.colorFalloff.c);

	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_FALLOFF_SCALE, renderDescription.falloffScale);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_THICKNESS_REFRACTION, renderDescription.thicknessRefraction);

	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_FRESNEL_BIAS, renderDescription.fresnelBias);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_FRESNEL_SCALE, renderDescription.fresnelScale);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_FRESNEL_POWER, renderDescription.fresnelPower);

	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_NORMAL_NOISE_WEIGTH, renderDescription.normalNoiseWeight);

	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_SCENE_MAP, sceneTexture);

	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_THICKNESS_MAP, thicknessTexture);
	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_FOAM_THICKNESS_MAP, foamThicknessTexture);

	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_NOISE_MAP, noiseTexture);

	Vector2 fluidBufferSize = bufferSize*lowResFactor;
	Vector2 invViewport(1.0f/fluidBufferSize.x, 1.0f/fluidBufferSize.y);
	Vector2 invCamera(-2.0f * invFocalLength * aspectRatio * invViewport.x, -2.0f * invFocalLength * invViewport.y);

	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_DEPTH_THRESHOLD, DEPTH_THRESHOLD);
	ShaderManager::Instance()->SetParameter2f(ShaderManager::SP_FLUID_INV_VIEWPORT, invViewport.comp);
	ShaderManager::Instance()->SetParameter2f(ShaderManager::SP_FLUID_INV_FOCAL_LENGTH, Vector2(invFocalLength*aspectRatio, invFocalLength));
	ShaderManager::Instance()->SetParameter2f(ShaderManager::SP_FLUID_INV_CAMERA, invCamera.comp);

	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_PARTICLE_SCALE, windowHeight / invFocalLength);
	ShaderManager::Instance()->SetParameter2f(ShaderManager::SP_FLUID_BUFFER_SIZE, bufferSize.comp);

	ShaderManager::Instance()->SetParameter4fv(ShaderManager::SP_FLUID_FOAM_BACK_COLOR, renderDescription.foamBackColor.c);
	ShaderManager::Instance()->SetParameter4fv(ShaderManager::SP_FLUID_FOAM_FRONT_COLOR, renderDescription.foamFrontColor.c);

	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_FOAM_FALLOFF_SCALE, renderDescription.foamFalloffScale);

	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_FOAM_DEPTH_THRESHOLD, renderDescription.foamDepthThreshold);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_FOAM_FRONT_FALLOFF_SCALE, renderDescription.foamFrontFalloffScale);
}


// -----------------------------------------------------------------------------
// -------------------- ScreenSpaceCurvature::UpdateMetaData -------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::UpdateMetaData(float deltaTime)
{
	unsigned int fluidBufferNum = fluid->GetFluidBufferNum();
	Fluid::FluidParticle* fluidBuffer = fluid->GetFluidBuffer();

	unsigned int fluidCreatedParticleIdsNum = fluid->GetFluidCreatedParticleIdsNum();
	const unsigned int* fluidCreatedParticleIds = fluid->GetFluidCreatedParticleIds();

	const float threshold = renderDescription.foamThreshold;
	const float lifetime = renderDescription.foamLifetime;

	// reset foam amount for new created particles
	unsigned int j;
	for (j=0; j<fluidCreatedParticleIdsNum; j++)
	{
		unsigned int id = fluidCreatedParticleIds[j];
		{
			fluidMetaData[id].foam = 0.0f;
			fluidMetaData[id].lifetime = 0.0f;
			fluidMetaData[id].timer = 0.0f;
			fluidMetaData[id].phase = FP_NONE;
		}
	}

	unsigned int i;
	for (i=0; i<fluidBufferNum; i++)
	{
		const float weberNumberScale = 0.00001f;

		// get particle
		const Fluid::FluidParticle& particle = fluidBuffer[i];
		float weberNumber = particle.velocity.Length()*particle.velocity.Length()*particle.density*weberNumberScale;

		switch (fluidMetaData[particle.id].phase)
		{
		case FP_NONE:			// do nothing as long as weber number is below threshold
			if (weberNumber >= threshold)
			{
				fluidMetaData[particle.id].foam = 0.0f;
				fluidMetaData[particle.id].lifetime = Math::RandomFloat(0.25f, 0.5f);
				fluidMetaData[particle.id].timer = fluidMetaData[particle.id].lifetime;
				fluidMetaData[particle.id].phase = FP_WATER_TO_FOAM;
			}
			break;
		case FP_WATER_TO_FOAM:	// fade-in foam
			fluidMetaData[particle.id].timer -= deltaTime;
			if (fluidMetaData[particle.id].timer <= 0.0f)
			{
				fluidMetaData[particle.id].foam = 1.0f;
				fluidMetaData[particle.id].lifetime = 0.0f;
				fluidMetaData[particle.id].timer = 0.0f;
				fluidMetaData[particle.id].phase = FP_FOAM;
			}
			else
			{
				fluidMetaData[particle.id].foam = 1.0f-fluidMetaData[particle.id].timer/fluidMetaData[particle.id].lifetime;
			}
			break;
		case FP_FOAM:			// as long as weber number is high keep foam at max
			if (weberNumber < threshold)
			{
				fluidMetaData[particle.id].foam = 1.0f;
				fluidMetaData[particle.id].lifetime = Math::Clamp(lifetime*Math::RandomFloat(0.5f, 1.5f), 0.0f, Math::MAX_FLOAT);
				fluidMetaData[particle.id].timer = fluidMetaData[particle.id].lifetime;
				fluidMetaData[particle.id].phase = FP_FOAM_TO_WATER;
			}
			break;
		case FP_FOAM_TO_WATER:	// fade back to water phase
			fluidMetaData[particle.id].timer -= deltaTime;
			if (fluidMetaData[particle.id].timer <= 0.0f)
			{
				fluidMetaData[particle.id].foam = 0.0f;
				fluidMetaData[particle.id].lifetime = 0.0f;
				fluidMetaData[particle.id].timer = 0.0f;
				fluidMetaData[particle.id].phase = FP_NONE;
			}
			else
			{
				fluidMetaData[particle.id].foam = fluidMetaData[particle.id].timer/fluidMetaData[particle.id].lifetime;
			}
			break;
		default:
			assert(false);
			break;
		}
	}
}


// -----------------------------------------------------------------------------
// ------------------------ ScreenSpaceCurvature::Render -----------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::Render(void)
{
	// Specify point sprite texture coordinate replacement mode for each texture unit
	glTexEnvf(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);

	// Copy particle data to vbo and activate the vbo
	UpdateVBO();

	// no blending for depth and filter passes
	glDisable(GL_BLEND);

	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	{
		RenderFoamDepth();
		RenderDepth();
	}

	glDepthMask(GL_FALSE);
	glDisable(GL_DEPTH_TEST);
	{
		RenderSmooth();
	}

	// enable additive blending for noise and thickness passes
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glBlendEquation(GL_FUNC_ADD);

	glEnable(GL_DEPTH_TEST);
	{
		if (renderDescription.useNoise)
			RenderNoise();

		RenderThickness();
		RenderFoamThickness();
	}

	// disable blending for final rendering
	glDisable(GL_BLEND);

	RenderComposition();

	// Disable vbo after rendering
	DisableVBO();

	// Disable point sprite texture coordinate replacement mode
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_FALSE);
}


// -----------------------------------------------------------------------------
// ------------------- ScreenSpaceCurvature::BeginRenderScene ------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::BeginRenderScene(void)
{
	SetRenderTarget(RENDER_TARGET_SCENE);
}


// -----------------------------------------------------------------------------
// -------------------- ScreenSpaceCurvature::EndRenderScene -------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::EndRenderScene(void)
{
	if (renderDescription.renderAABB)
		RenderBoundingBoxes();

	glBindTexture(GL_TEXTURE_2D, 0);
	SetRenderTarget(RENDER_TARGET_DISABLED);
}


// -----------------------------------------------------------------------------
// --------------------- ScreenSpaceCurvature::Exit ----------------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::Exit(void)
{
	glDeleteTextures(1, &sceneTexture);

	glDeleteTextures(1, &thicknessTexture);
	glDeleteTextures(1, &foamThicknessTexture);
	
	glDeleteTextures(1, &depthTexture);
	glDeleteTextures(1, &foamDepthTexture);

#if USE_DOWNSAMPLE_SHIFT
	glDeleteTextures(DOWNSAMPLE_SHIFT, downsampleDepthTexture);
	glDeleteTextures(DOWNSAMPLE_SHIFT, downsampleFoamDepthTexture);
#endif

	glDeleteTextures(1, &smoothedDepthTexture);
	glDeleteTextures(1, &noiseTexture);
	glDeleteTextures(1, &resultTexture);

	glDeleteTextures(1, &perlinNoiseTexture);
	glDeleteTextures(1, &foamPerlinNoiseTexture);

	glDeleteFramebuffersEXT(1, &frameBuffer);
	glDeleteRenderbuffersEXT(1, &renderBuffer);
	glDeleteRenderbuffersEXT(1, &depthRenderBuffer);

	delete[] vboStartIndices;
	vboStartIndices = NULL;

	delete[] vboIndexCount;
	vboIndexCount = NULL;

	if (vbo > 0)
	{
		glDeleteBuffersARB(1, &vbo);
		vbo = 0;
	}

	delete[] fluidMetaData;
	fluidMetaData = NULL;

	glDeleteQueriesARB(1, &occQuery);
}


// -----------------------------------------------------------------------------
// ----------------- ScreenSpaceCurvature::SetWindowSize -----------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::SetWindowSize(int width, int height)
{
	windowWidth = width;
	windowHeight = height;
	aspectRatio = (float) windowWidth / (float) windowHeight;

	invFocalLength = Math::Tan(fieldOfView*Math::DEG_TO_RAD*0.5f);

	Exit();
	Init();
}


// -----------------------------------------------------------------------------
// ---------------------- ScreenSpaceCurvature::UpdateVBO ----------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::UpdateVBO(void)
{
	unsigned int numPackets = fluid->GetFluidNumPackets();
	const NxFluidPacket* packets = fluid->GetFluidPackets();

	visiblePacketCount = 0;

	// determine visible pakcets
	unsigned int i;
	for (i=0; i<numPackets; i++)
	{
		if (FrustumAABBIntersect(frustumPlanes,
			Vector3(packets[i].aabb.min.x, packets[i].aabb.min.y, packets[i].aabb.min.z),
			Vector3(packets[i].aabb.max.x, packets[i].aabb.max.y, packets[i].aabb.max.z)))
		{
			vboStartIndices[visiblePacketCount] = packets[i].firstParticleIndex;
			vboIndexCount[visiblePacketCount] = packets[i].numParticles;
			visiblePacketCount++;
		}
	}

	/// update foam (sync PhysiX simulation data and fluid meta data)
	unsigned int fluidBufferNum = fluid->GetFluidBufferNum();
	Fluid::FluidParticle* fluidBuffer = fluid->GetFluidBuffer();

	for (i=0; i<fluidBufferNum; i++)
	{
		Fluid::FluidParticle& particle = fluidBuffer[i];
		particle.foam = fluidMetaData[particle.id].foam;
	}

	// activate VBO
	const unsigned int stride = sizeof(Fluid::FluidParticle);
	const unsigned int offsetForDensity = sizeof(GLfloat) * 3;
	const unsigned int offsetForVelocity = sizeof(GLfloat) * 4;
	const unsigned int offsetLifeTime = sizeof(GLfloat) * 7;
	const unsigned int offsetFoam = sizeof(GLfloat) * 8;

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, fluid->GetMaxParticles()*sizeof(Fluid::FluidParticle), NULL, GL_STREAM_DRAW_ARB);

	// copy new particle data
	Fluid::FluidParticle* ptr = (Fluid::FluidParticle*)glMapBufferARB(GL_ARRAY_BUFFER_ARB, GL_WRITE_ONLY_ARB);
	if(ptr)
	{
		memcpy(ptr, fluidBuffer, fluidBufferNum*sizeof(Fluid::FluidParticle));

		glUnmapBufferARB(GL_ARRAY_BUFFER_ARB);
	}

	// setup pointers for rendering
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, stride, BUFFER_OFFSET(0));

	glActiveTextureARB(GL_TEXTURE1);
	glClientActiveTextureARB(GL_TEXTURE1);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(1, GL_FLOAT, stride, BUFFER_OFFSET(offsetForDensity));

	glActiveTextureARB(GL_TEXTURE2);
	glClientActiveTextureARB(GL_TEXTURE2);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(3, GL_FLOAT, stride, BUFFER_OFFSET(offsetForVelocity));

	glActiveTextureARB(GL_TEXTURE3);
	glClientActiveTextureARB(GL_TEXTURE3);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(1, GL_FLOAT, stride, BUFFER_OFFSET(offsetLifeTime));

	glActiveTextureARB(GL_TEXTURE4);
	glClientActiveTextureARB(GL_TEXTURE4);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(1, GL_FLOAT, stride, BUFFER_OFFSET(offsetFoam));
}


// -----------------------------------------------------------------------------
// ------------------- ScreenSpaceCurvature::RenderFoamDepth -------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::RenderFoamDepth(void)
{
	SetRenderTarget(RENDER_TARGET_FOAM_DEPTH);
	glClearColor(-10000.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// glDepthMask(GL_TRUE);
	{
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_DENSITY_THRESHOLD, renderDescription.densityThreshold);
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_PARTICLE_SIZE, renderDescription.particleSize*2.0f);
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_FLUID, "FluidFoamDepth");
		{
			RenderParticles();
		}
		ShaderManager::Instance()->DisableShader();
	}

	SetRenderTarget(RENDER_TARGET_DISABLED);
}


// -----------------------------------------------------------------------------
// --------------------- ScreenSpaceCurvature::RenderDepth ---------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::RenderDepth(void)
{
	SetRenderTarget(RENDER_TARGET_DEPTH);
	glClearColor(-10000.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//glDisable(GL_BLEND);
	{
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_DENSITY_THRESHOLD, renderDescription.densityThreshold);
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_PARTICLE_SIZE, renderDescription.particleSize);
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_FLUID, "FluidDepth");
		{
			RenderParticles();
		}
		ShaderManager::Instance()->DisableShader();
	}
	SetRenderTarget(RENDER_TARGET_DISABLED);
}


// -----------------------------------------------------------------------------
// --------------------- ScreenSpaceCurvature::RenderSmooth --------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::RenderSmooth(void)
{
	// glDisable(GL_DEPTH_TEST);
	{
		scaleDownWidth = windowWidth;
		scaleDownHeight = windowHeight;

		unsigned int depthSrc = depthTexture;

#ifdef USE_DOWNSAMPLE_SHIFT
		int i;
		for(i=0; i<DOWNSAMPLE_SHIFT; i++)
		{
			ScaleDownTexture(downsampleDepthTexture[i], depthSrc);
			depthSrc = downsampleDepthTexture[i];
		}
#endif

		unsigned int bounceTexture[2];
		bounceTexture[0] = depthSrc;
		bounceTexture[1] = smoothedDepthTexture;

		unsigned int idx = SmoothTexture(bounceTexture);
		currentDepthSource = bounceTexture[idx];
	}

	{
		scaleDownWidth = windowWidth;
		scaleDownHeight = windowHeight;

		unsigned int depthSrc = foamDepthTexture;

#if USE_DOWNSAMPLE_SHIFT
		int i;
		for(i=0; i<DOWNSAMPLE_SHIFT; i++)
		{
			ScaleDownTexture(downsampleFoamDepthTexture[i], depthSrc);
			depthSrc = downsampleFoamDepthTexture[i];
		}
#endif

		currentFoamDepthSource = depthSrc;
	}
}


// -----------------------------------------------------------------------------
// --------------------- ScreenSpaceCurvature::RenderNoise ---------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::RenderNoise(void)
{
	SetRenderTarget(RENDER_TARGET_NOISE);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f); 
	glClear(GL_COLOR_BUFFER_BIT);

	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_PERLIN_MAP, perlinNoiseTexture);
	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_DEPTH_MAP, currentDepthSource);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_LOW_RES_FACTOR, lowResFactor);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_NOISE_DEPTH_FALLOFF, renderDescription.noiseDepthFalloff);

	// use linear depth interpolation for noise purposes
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, currentDepthSource);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

	// glDepthMask(GL_FALSE); glBlendFunc(GL_ONE, GL_ONE); glBlendEquation(GL_FUNC_ADD);
	{
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_PARTICLE_SIZE, renderDescription.particleSize*2.0f);
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_FLUID, "FluidNoise");
		{
			RenderParticles();
		}
		ShaderManager::Instance()->DisableShader();
	}

	// reset interpolation
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, currentDepthSource);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

	SetRenderTarget(RENDER_TARGET_DISABLED);
}


// -----------------------------------------------------------------------------
// ------------------- ScreenSpaceCurvature::RenderThickness -------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::RenderThickness(void)
{
	SetRenderTarget(RENDER_TARGET_THICKNESS);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// glDepthMask(GL_FALSE); glBlendFunc(GL_ONE, GL_ONE); glBlendEquation(GL_FUNC_ADD);
	{
		Vector2 scaledDownSize(scaleDownWidth, scaleDownHeight);
		ShaderManager::Instance()->SetParameter2f(ShaderManager::SP_FLUID_SCALED_DOWN_SIZE, scaledDownSize.comp);
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_LOW_RES_FACTOR, 1.0f/lowResFactor);
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_PARTICLE_SIZE, renderDescription.particleSize*2.0f);
		ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_DEPTH_MAP, currentFoamDepthSource);
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_FACE_SCALE, renderDescription.fluidThicknessScale);

		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_FLUID, "FluidThickness");
		{
			RenderParticles();
		}
		ShaderManager::Instance()->DisableShader();
	}

	SetRenderTarget(RENDER_TARGET_DISABLED);
}


// -----------------------------------------------------------------------------
// ----------------- ScreenSpaceCurvature::RenderFoamThickness -----------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::RenderFoamThickness(void)
{
	SetRenderTarget(RENDER_TARGET_FOAM_THICKNESS);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// glDepthMask(GL_FALSE); glBlendFunc(GL_ONE, GL_ONE); glBlendEquation(GL_FUNC_ADD);
	{
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_LOW_RES_FACTOR, 1.0f/lowResFactor);
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_PARTICLE_SIZE, renderDescription.particleSize*3.0f);
		ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_DEPTH_MAP, currentFoamDepthSource);
		ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_PERLIN_MAP, foamPerlinNoiseTexture);
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_FACE_SCALE, renderDescription.foamThicknessScale);

		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_FLUID, "FluidFoamThickness");
		{
			RenderParticles();
		}
		ShaderManager::Instance()->DisableShader();
	}

	SetRenderTarget(RENDER_TARGET_DISABLED);
}


// -----------------------------------------------------------------------------
// ------------------ ScreenSpaceCurvature::RenderComposition ------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::RenderComposition(void)
{
	float specular[3];
	specular[0] = renderDescription.specularColor.r;
	specular[1] = renderDescription.specularColor.g;
	specular[2] = renderDescription.specularColor.b;

	Vector2 fluidBufferSize = bufferSize*lowResFactor;
	Vector2 invViewport(1.0f/fluidBufferSize.x, 1.0f/fluidBufferSize.y);

	float icx = 2.0f * invFocalLength * aspectRatio * invViewport.x;
	float icy = 2.0f * invFocalLength * invViewport.y;
	Vector3 fluidCamera(icy, icx, icx*icy);

	ShaderManager::Instance()->SetParameter3fv(ShaderManager::SP_FLUID_LIGHT_POS_EYE_SPACE, fluidLightPosition.comp);
	ShaderManager::Instance()->SetParameter3fv(ShaderManager::SP_FLUID_CAMERA, fluidCamera.comp);

	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_DEPTH_MAP, currentDepthSource);
	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_FOAM_DEPTH_MAP, currentFoamDepthSource);
	
	ShaderManager::Instance()->SetParameter3fv(ShaderManager::SP_FLUID_SPECULAR_COLOR, specular);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_SHININESS, renderDescription.specularShininess);

	SetRenderTarget(RENDER_TARGET_RESULT);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_DEPTH_TEST);
	{
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_FLUID, "FluidComposition");
		{
			RenderQuad(fluidBufferSize.x, fluidBufferSize.y);
		}
		ShaderManager::Instance()->DisableShader();
	}
	glEnable(GL_DEPTH_TEST);

	// finally, render the spray
	RenderSpray();

	SetRenderTarget(RENDER_TARGET_DISABLED);
}


// -----------------------------------------------------------------------------
// --------------------- ScreenSpaceCurvature::RenderSpray ---------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::RenderSpray(void)
{
	// spray color
	glColor4f(renderDescription.sprayColor.r*renderDescription.sprayColor.a,
		renderDescription.sprayColor.g*renderDescription.sprayColor.a,
		renderDescription.sprayColor.b*renderDescription.sprayColor.a,
		1.0f);

	glDepthMask(GL_FALSE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glBlendEquation(GL_FUNC_ADD);
	{
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_PARTICLE_SIZE, renderDescription.particleSize*0.5f);
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_FLUID, "FluidSpray");
		{
			RenderParticles();
		}
		ShaderManager::Instance()->DisableShader();
	}
	glDisable(GL_BLEND);
	glDepthMask(GL_TRUE);
}


// -----------------------------------------------------------------------------
// ---------------------- ScreenSpaceCurvature::DisableVBO ---------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::DisableVBO(void) const
{
	glDisableClientState(GL_VERTEX_ARRAY);

	glClientActiveTexture(GL_TEXTURE1);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glClientActiveTexture(GL_TEXTURE2);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glClientActiveTexture(GL_TEXTURE3);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glClientActiveTexture(GL_TEXTURE4);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, NULL);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, NULL);
}


// -----------------------------------------------------------------------------
// ------------------- ScreenSpaceCurvature::RenderParticles -------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::RenderParticles(void)
{
	// render point sprites
	glEnable(GL_POINT_SPRITE_ARB);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

	glMultiDrawArrays(GL_POINTS, vboStartIndices, vboIndexCount, visiblePacketCount);

	glDisable(GL_POINT_SPRITE_ARB);
	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
}


// -----------------------------------------------------------------------------
// ------------------- ScreenSpaceCurvature::SetRenderTarget -------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::SetRenderTarget(RenderTarget target)
{
	int viewportWidth = 0;
	int viewportHeight = 0;

	switch(target) {
	case RENDER_TARGET_DISABLED:
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
		viewportWidth = windowWidth;
		viewportHeight = windowHeight;
		break;

	case RENDER_TARGET_SCENE:
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frameBuffer);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, sceneTexture, 0);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, renderBuffer);
		viewportWidth = windowWidth;
		viewportHeight = windowHeight;
		break;

	case RENDER_TARGET_FOAM_DEPTH:
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frameBuffer);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, foamDepthTexture, 0);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthRenderBuffer);
		viewportWidth = windowWidth;
		viewportHeight = windowHeight;
		break;

	case RENDER_TARGET_DEPTH:
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frameBuffer);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, depthTexture, 0);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthRenderBuffer);
		viewportWidth = windowWidth;
		viewportHeight = windowHeight;
		break;

	case RENDER_TARGET_THICKNESS:
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frameBuffer);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, thicknessTexture, 0);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_RECTANGLE_ARB, NULL, 0);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, renderBuffer);
		viewportWidth = windowWidth;
		viewportHeight = windowHeight;
		break;

	case RENDER_TARGET_FOAM_THICKNESS:
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frameBuffer);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, foamThicknessTexture, 0);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_RECTANGLE_ARB, NULL, 0);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, renderBuffer);
		viewportWidth = windowWidth;
		viewportHeight = windowHeight;
		break;

	case RENDER_TARGET_NOISE:
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frameBuffer);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, noiseTexture, 0);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, renderBuffer);
		viewportWidth = windowWidth;
		viewportHeight = windowHeight;
		break;

	case RENDER_TARGET_RESULT:
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frameBuffer);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, resultTexture, 0);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, renderBuffer);
		viewportWidth = windowWidth;
		viewportHeight = windowHeight;
		break;
	}

	glViewport(0, 0, viewportWidth, viewportHeight);
}


// -----------------------------------------------------------------------------
// ---------------------- ScreenSpaceCurvature::RenderQuad ---------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::RenderQuad(float u, float v) const
{
	glBegin(GL_QUADS);
	{
		glTexCoord2f(0.0f, 0.0f);
		glVertex3f (-1.0f, -1.0f, 0.0f);

		glTexCoord2f(u, 0.0f);
		glVertex3f (1.0f, -1.0f, 0.0f);

		glTexCoord2f(u, v);
		glVertex3f (1.0f, 1.0f, 0.0f);

		glTexCoord2f(0.0f, v);
		glVertex3f (-1.0f, 1.0f, 0.0f);
	}
	glEnd();
}


// -----------------------------------------------------------------------------
// ----------------- ScreenSpaceCurvature::RenderBoundingBoxes -----------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::RenderBoundingBoxes(void)
{
	unsigned int numPackets = fluid->GetFluidNumPackets();
	const NxFluidPacket* packets = fluid->GetFluidPackets();

	unsigned int i;
	for (i=0; i<numPackets; i++)
	{
		if (!FrustumAABBIntersect(frustumPlanes,
			Vector3(packets[i].aabb.min.x, packets[i].aabb.min.y, packets[i].aabb.min.z),
			Vector3(packets[i].aabb.max.x, packets[i].aabb.max.y, packets[i].aabb.max.z)))
		{
			glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
		}
		else
		{
			glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
		}

		glBegin(GL_LINES);
		{
			// bottom
			glVertex3f(packets[i].aabb.min.x, packets[i].aabb.min.y, packets[i].aabb.min.z);
			glVertex3f(packets[i].aabb.min.x, packets[i].aabb.min.y, packets[i].aabb.max.z);

			glVertex3f(packets[i].aabb.max.x, packets[i].aabb.min.y, packets[i].aabb.min.z);
			glVertex3f(packets[i].aabb.max.x, packets[i].aabb.min.y, packets[i].aabb.max.z);

			glVertex3f(packets[i].aabb.min.x, packets[i].aabb.min.y, packets[i].aabb.min.z);
			glVertex3f(packets[i].aabb.max.x, packets[i].aabb.min.y, packets[i].aabb.min.z);

			glVertex3f(packets[i].aabb.min.x, packets[i].aabb.min.y, packets[i].aabb.max.z);
			glVertex3f(packets[i].aabb.max.x, packets[i].aabb.min.y, packets[i].aabb.max.z);

			// top
			glVertex3f(packets[i].aabb.min.x, packets[i].aabb.max.y, packets[i].aabb.min.z);
			glVertex3f(packets[i].aabb.min.x, packets[i].aabb.max.y, packets[i].aabb.max.z);

			glVertex3f(packets[i].aabb.max.x, packets[i].aabb.max.y, packets[i].aabb.min.z);
			glVertex3f(packets[i].aabb.max.x, packets[i].aabb.max.y, packets[i].aabb.max.z);

			glVertex3f(packets[i].aabb.min.x, packets[i].aabb.max.y, packets[i].aabb.min.z);
			glVertex3f(packets[i].aabb.max.x, packets[i].aabb.max.y, packets[i].aabb.min.z);

			glVertex3f(packets[i].aabb.min.x, packets[i].aabb.max.y, packets[i].aabb.max.z);
			glVertex3f(packets[i].aabb.max.x, packets[i].aabb.max.y, packets[i].aabb.max.z);

			// side
			glVertex3f(packets[i].aabb.min.x, packets[i].aabb.min.y, packets[i].aabb.min.z);
			glVertex3f(packets[i].aabb.min.x, packets[i].aabb.max.y, packets[i].aabb.min.z);

			glVertex3f(packets[i].aabb.min.x, packets[i].aabb.min.y, packets[i].aabb.max.z);
			glVertex3f(packets[i].aabb.min.x, packets[i].aabb.max.y, packets[i].aabb.max.z);

			glVertex3f(packets[i].aabb.max.x, packets[i].aabb.min.y, packets[i].aabb.min.z);
			glVertex3f(packets[i].aabb.max.x, packets[i].aabb.max.y, packets[i].aabb.min.z);

			glVertex3f(packets[i].aabb.max.x, packets[i].aabb.min.y, packets[i].aabb.max.z);
			glVertex3f(packets[i].aabb.max.x, packets[i].aabb.max.y, packets[i].aabb.max.z);

		}
		glEnd();
	}
}


// -----------------------------------------------------------------------------
// ------------------- ScreenSpaceCurvature::ScaleDownTexture ------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::ScaleDownTexture(unsigned int dest, unsigned int src)
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frameBuffer);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, dest, 0);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_RECTANGLE_ARB, NULL, 0);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, 0);

	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_DEPTH_MAP, src);
	ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_FLUID, "FluidDownsample");
	{
		glViewport(0, 0, scaleDownWidth/2, scaleDownHeight/2);
		RenderQuad(scaleDownWidth, scaleDownHeight);
	}
	ShaderManager::Instance()->DisableShader();
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	scaleDownWidth *= 0.5f;
	scaleDownHeight *= 0.5f;
}


// -----------------------------------------------------------------------------
// -------------------- ScreenSpaceCurvature::SmoothTexture --------------------
// -----------------------------------------------------------------------------
unsigned int ScreenSpaceCurvature::SmoothTexture(unsigned int tex[2])
{
	glClearColor(-10000.0f, 0.0f, 0.0f, 0.0f);

	const float icx = 2.0f * invFocalLength * aspectRatio / (float)scaleDownWidth;
	const float icy = 2.0f * invFocalLength / (float)scaleDownHeight;
	const Vector3 fluidCamera(icy, icx, icx*icy);

	const float eulerIterationFactor = (renderDescription.worldSpaceKernelRadius * (float)scaleDownWidth) / (invFocalLength * 2.0f);

	int srcTex = 0;

	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_EPSILON, EPSILON);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_BLUR_DEPTH_FALLOFF, BLUR_DEPTH_FALLOFF);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_THRESHOLDMIN, THRESHOLD_MIN);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_EULER_ITERATION_FACTOR, eulerIterationFactor);
	ShaderManager::Instance()->SetParameter3fv(ShaderManager::SP_FLUID_CAMERA, fluidCamera.comp);

	bool breakLoop = false;
	bool doQuery = false;

	const unsigned int queryInterval = 5;

	int i;
	for (i=0; i<256 && !breakLoop; i++)
	{
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frameBuffer);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, tex[1 - srcTex], 0);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, 0);

		if (i==0)
			glClear(GL_COLOR_BUFFER_BIT);

		if (i%queryInterval == (queryInterval-1))
			doQuery = true;

		if (doQuery)
			glBeginQueryARB(GL_SAMPLES_PASSED_ARB, occQuery);

		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_CURRENT_ITERATION, i);

		ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_DEPTH_MAP, tex[srcTex]);
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_FLUID, "FluidSmoothing");
		{
			RenderQuad(scaleDownWidth, scaleDownHeight);
		}
		ShaderManager::Instance()->DisableShader();

		if (doQuery)
			glEndQueryARB(GL_SAMPLES_PASSED_ARB);

		srcTex = 1 - srcTex;

		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, tex[1 - srcTex], 0);

		ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_DEPTH_MAP, tex[srcTex]);
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_FLUID, "FluidSmoothingPassThrough");
		{
			RenderQuad(scaleDownWidth, scaleDownHeight);
		}
		ShaderManager::Instance()->DisableShader();

		if (doQuery)
		{
			unsigned int done = 0;
			while (!done)
				glGetQueryObjectuivARB(occQuery, GL_QUERY_RESULT_AVAILABLE_ARB, &done);

			GLuint fragmentCount;
			glGetQueryObjectuivARB(occQuery , GL_QUERY_RESULT_ARB, &fragmentCount);

			if (fragmentCount == 0)
				breakLoop = true;

			doQuery = false;
		}

		srcTex = 1 - srcTex;
	}

	currentIterationCount = i;
	return srcTex;
}


// -----------------------------------------------------------------------------
// ----------------- ScreenSpaceCurvature::FrustumAABBIntersect ----------------
// -----------------------------------------------------------------------------
bool ScreenSpaceCurvature::FrustumAABBIntersect(const Math::FrustumPlane* planes, const Vector3& mins, const Vector3& maxs) const
{ 
	Vector3 vec;

	unsigned int i;
	for (i=0; i<6; i++)
	{ 
		// X axis
		if (planes[i].normal.x > 0)
			vec.x = maxs.x;
		else
			vec.x = mins.x;

		// Y axis
		if (planes[i].normal.y > 0)
			vec.y = maxs.y;
		else
			vec.y = mins.y;

		// Z axis 
		if (planes[i].normal.z > 0)
			vec.z = maxs.z;
		else
			vec.z = mins.z;

		if (planes[i].normal.DotProduct(vec) + planes[i].d < 0.0f)
			return false;
	}

	return true;
}



// -----------------------------------------------------------------------------
// ---------------- ScreenSpaceCurvature::CheckFrameBufferState ----------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::CheckFrameBufferState(void) const
{
	GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
	switch( status )
	{
	case GL_FRAMEBUFFER_COMPLETE_EXT:
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
		assert(false);
		break;
	default:
		assert(false);
	}
}


// -----------------------------------------------------------------------------
// -------------------- ScreenSpaceCurvature::CreateTexture --------------------
// -----------------------------------------------------------------------------
unsigned int ScreenSpaceCurvature::CreateTexture(GLenum target, int w, int h, unsigned int internalformat, GLenum format)
{
	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(target, texture);

	glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_FLOAT, 0);
	return texture;
}


// -----------------------------------------------------------------------------
// ------------------ ScreenSpaceCurvature::BuildNoiseTexture ------------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::BuildNoiseTexture(void)
{
	unsigned char* texels = (BYTE *)malloc(NOISE_WIDTH * NOISE_HEIGHT * NOISE_DEPTH * BYTES_PER_TEXEL);

	unsigned int r, s, t;

	Perlin* perlin = new Perlin();
	perlin->Initialize(1);

	for (r=0; r<NOISE_DEPTH; r++)
	{
		for (s=0; s<NOISE_WIDTH; s++)
		{
			for (t=0; t<NOISE_HEIGHT; t++)
			{
				float x = Math::Clamp((float)s/(float)(NOISE_WIDTH-1), 0.0f, 1.0f)*16.0f;
				float y = Math::Clamp((float)t/(float)(NOISE_HEIGHT-1), 0.0f, 1.0f)*16.0f;
				float z = Math::Clamp((float)r/(float)(NOISE_DEPTH-1), 0.0f, 1.0f)*16.0f;

				float noiseValue = perlin->Noise3(x, y, z)*0.5f + 0.5f;

				texels[TEXEL3(s, t, r)] = (int)(noiseValue*255.0f);
				texels[TEXEL3(s, t, r)+1] = (int)(noiseValue*255.0f);
				texels[TEXEL3(s, t, r)+2] = (int)(noiseValue*255.0f);
			}
		}
	}

	glGenTextures(1, &perlinNoiseTexture);	
	glBindTexture(GL_TEXTURE_3D, perlinNoiseTexture);	
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB8, NOISE_WIDTH, NOISE_HEIGHT, NOISE_DEPTH, 0, GL_RGB, GL_UNSIGNED_BYTE, texels);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	glBindTexture(GL_TEXTURE_3D, NULL);

	delete texels;
	delete perlin;
}


// -----------------------------------------------------------------------------
// ---------------- ScreenSpaceCurvature::BuildFoamNoiseTexture ----------------
// -----------------------------------------------------------------------------
void ScreenSpaceCurvature::BuildFoamNoiseTexture(void)
{
	unsigned char* texels = (BYTE *)malloc(FOAM_NOISE_WIDTH * FOAM_NOISE_HEIGHT * FOAM_NOISE_DEPTH * BYTES_PER_TEXEL);

	unsigned int r, s, t;

	Perlin* perlin = new Perlin();
	perlin->Initialize(1);

	float minNoiseValue =  Math::MAX_FLOAT;
	float maxNoiseValue = -Math::MAX_FLOAT;

	for (r=0; r<FOAM_NOISE_DEPTH; r++)
	{
		for (s=0; s<FOAM_NOISE_WIDTH; s++)
		{
			for (t=0; t<FOAM_NOISE_HEIGHT; t++)
			{
				float x = Math::Clamp((float)s/(float)(FOAM_NOISE_WIDTH-1), 0.0f, 1.0f)*4.0f;
				float y = Math::Clamp((float)t/(float)(FOAM_NOISE_HEIGHT-1), 0.0f, 1.0f)*4.0f;
				float z = Math::Clamp((float)r/(float)(FOAM_NOISE_DEPTH-1), 0.0f, 1.0f)*4.0f;

				float noiseValue = perlin->Noise3(x, y, z)*0.5f + 0.5f;

				minNoiseValue = Math::Min(minNoiseValue, noiseValue*255.0f);
				maxNoiseValue = Math::Max(maxNoiseValue, noiseValue*255.0f);

				texels[FOAM_TEXEL3(s, t, r)] = (int)(noiseValue*255.0f);
				texels[FOAM_TEXEL3(s, t, r)+1] = (int)(noiseValue*255.0f);
				texels[FOAM_TEXEL3(s, t, r)+2] = (int)(noiseValue*255.0f);
			}
		}
	}

	for (r=0; r<FOAM_NOISE_DEPTH; r++)
	{
		for (s=0; s<FOAM_NOISE_WIDTH; s++)
		{
			for (t=0; t<FOAM_NOISE_HEIGHT; t++)
			{
				float noiseValue = (float)texels[FOAM_TEXEL3(s, t, r)];
				noiseValue = (noiseValue - minNoiseValue)*(255.0f/(maxNoiseValue-minNoiseValue));

				texels[FOAM_TEXEL3(s, t, r)] = (int)(noiseValue);
				texels[FOAM_TEXEL3(s, t, r)+1] = (int)(noiseValue);
				texels[FOAM_TEXEL3(s, t, r)+2] = (int)(noiseValue);
			}
		}
	}

	glGenTextures(1, &foamPerlinNoiseTexture);	
	glBindTexture(GL_TEXTURE_3D, foamPerlinNoiseTexture);	
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB8, FOAM_NOISE_WIDTH, FOAM_NOISE_HEIGHT, FOAM_NOISE_DEPTH, 0, GL_RGB, GL_UNSIGNED_BYTE, texels);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	glBindTexture(GL_TEXTURE_3D, NULL);

	delete texels;
	delete perlin;
}
