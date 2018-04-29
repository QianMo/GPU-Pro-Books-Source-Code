#include <blossom_engine/blossom_engine.hpp>

#ifdef WIN32
	#if defined(DEBUG) | defined(_DEBUG)
		#include <crtdbg.h>
	#endif
#endif



string localDataDir = "../../data/";

CCamera camera;
vec4 rotationVectors[11][11];

int shadowMapSize = 512;

CRenderTarget shadowMap32RT1, shadowMap32RT2; // one-channel (SSM, ESM)
CRenderTarget shadowMap64RT1, shadowMap64RT2; // two-channel (VSM, EVSM)
CDepthStencilSurface shadowMapDSS;
CVertexDeclaration screenVD;
CVertexBuffer screenVB;
CVertexShader screenQuadVS, shadowMapVS, lightVS;
CPixelShader shadowMapPS[4], shadowMapHorizontalBlurPS, shadowMapVerticalBlurPS, lightPS[5];
CTexture texture;
CMesh mesh;
CRenderableMesh renderableMesh;

struct ScreenVertex
{
	vec3 position;
	vec2 texCoord;
};



void initCamera()
{
	camera.horizontalAngle = -PI/6.0f;
	camera.verticalAngle = -PI/5.0f;
	camera.updateFixed(vec3(-8.0f, 8.0f, 8.0f), vec3());
}



void updateCamera(float deltaTime)
{
	vec3 eye;

	float speed = 0.02f;
	if (Application.isKeyPressed(SDLK_LSHIFT))
		speed = 0.08f;

	eye = camera.getEye();
	if (Application.isKeyPressed(SDLK_w))
		eye += speed * deltaTime * camera.getForwardVector();
	if (Application.isKeyPressed(SDLK_s))
		eye -= speed * deltaTime * camera.getForwardVector();
	if (Application.isKeyPressed(SDLK_a))
		eye -= speed * deltaTime * camera.getRightVector();
	if (Application.isKeyPressed(SDLK_d))
		eye += speed * deltaTime * camera.getRightVector();

	camera.updateFree(eye);
}



void mouseMotionFunction()
{
	camera.horizontalAngle -= Application.getMouseRelX() / 1000.0f;
	camera.verticalAngle -= Application.getMouseRelY() / 1000.0f;
}



void init()
{
	initCamera();

	for (int x = -5; x <= 5; x++)
	{
		for (int z = -5; z <= 5; z++)
		{
			vec3 rotationVector = (2.0f*vec3(randf(), randf(), randf()) -  vec3(1.0f, 1.0f, 1.0f)).getNormalized();
			rotationVectors[x+5][z+5] = vec4(rotationVector.x, rotationVector.y, rotationVector.z, randf(0.0f, 2.0f*PI));
		}
	}



	Renderer.setSamplerFiltering_d3d9(0, SamplerFiltering::Point, SamplerFiltering::Point, SamplerFiltering::None);
	Renderer.setSamplerFiltering_d3d9(1, SamplerFiltering::Linear, SamplerFiltering::Linear, SamplerFiltering::Linear);
	Renderer.setSamplerFiltering_d3d9(2, SamplerFiltering::Linear, SamplerFiltering::Linear, SamplerFiltering::Linear);
	Renderer.setSamplerAddressing_d3d9(0, SamplerAddressing::Border);
	Renderer.setSamplerAddressing_d3d9(1, SamplerAddressing::Border);
	Renderer.setSamplerBorderColor_d3d9(0, vec4(0.0f, 0.0f, 0.0f, 0.0f));
	Renderer.setSamplerBorderColor_d3d9(1, vec4(0.0f, 0.0f, 0.0f, 0.0f));



	shadowMap32RT1.create(shadowMapSize, shadowMapSize, 1, TextureUsage::RenderTarget, TextureFormat::R32F);
	shadowMap32RT2.create(shadowMapSize, shadowMapSize, 1, TextureUsage::RenderTarget, TextureFormat::R32F);
	shadowMap64RT1.create(shadowMapSize, shadowMapSize, 1, TextureUsage::RenderTarget, TextureFormat::G32R32F);
	shadowMap64RT2.create(shadowMapSize, shadowMapSize, 1, TextureUsage::RenderTarget, TextureFormat::G32R32F);

	shadowMapDSS.create(shadowMapSize, shadowMapSize, 1, TextureUsage::DepthStencil, TextureFormat::D24);

	screenVD.create();
	screenVD.addVertexElement(0, VertexElementType::Float3, VertexElementSemantic::Position, 0);
	screenVD.addVertexElement(12, VertexElementType::Float2, VertexElementSemantic::TexCoord, 0);
	screenVD.build();

	screenVB.create(4*sizeof(ScreenVertex));

	ScreenVertex *vertices;
	screenVB.map((void*&)vertices);
	{
		vertices[0].position = vec3(-1.0f, 1.0f, 0.0f);
		vertices[1].position = vec3(-1.0f, -1.0f, 0.0f);
		vertices[2].position = vec3(1.0f, 1.0f, 0.0f);
		vertices[3].position = vec3(1.0f, -1.0f, 0.0f);

		vertices[0].texCoord = vec2(0.0f, 0.0f);
		vertices[1].texCoord = vec2(0.0f, 1.0f);
		vertices[2].texCoord = vec2(1.0f, 0.0f);
		vertices[3].texCoord = vec2(1.0f, 1.0f);
	}
	screenVB.unmap();

	screenQuadVS.createFromFile(localDataDir + "screen_quad.hlsl_vs");
	shadowMapVS.createFromFile(localDataDir + "shadow_map.hlsl_vs");
	lightVS.createFromFile(localDataDir + "light.hlsl_vs");

	shadowMapPS[0].createFromFile(localDataDir + "shadow_map.hlsl_ps", "SSM_ID");
	shadowMapPS[1].createFromFile(localDataDir + "shadow_map.hlsl_ps", "VSM_ID");
	shadowMapPS[2].createFromFile(localDataDir + "shadow_map.hlsl_ps", "ESM_ID");
	shadowMapPS[3].createFromFile(localDataDir + "shadow_map.hlsl_ps", "EVSM_ID");
	shadowMapHorizontalBlurPS.createFromFile(localDataDir + "shadow_map_horizontal_blur.hlsl_ps");
	shadowMapVerticalBlurPS.createFromFile(localDataDir + "shadow_map_vertical_blur.hlsl_ps");
	lightPS[0].createFromFile(localDataDir + "light.hlsl_ps", "SSM_ID");
	lightPS[1].createFromFile(localDataDir + "light.hlsl_ps", "VSM_ID");
	lightPS[2].createFromFile(localDataDir + "light.hlsl_ps", "VSM2_ID");
	lightPS[3].createFromFile(localDataDir + "light.hlsl_ps", "ESM_ID");
	lightPS[4].createFromFile(localDataDir + "light.hlsl_ps", "EVSM_ID");

	texture.createFromFile(localDataDir + "rocks.png", 0);

	mesh.importASE(localDataDir + "mesh.ASE");

	renderableMesh.setMesh(mesh);
	renderableMesh.createBuffers();
	renderableMesh.updateBuffers();
}



void renderSceneGeometry(const mtx& viewProjTransform, const mtx& lightViewProjTransform)
{
	Renderer.setVertexDeclaration(GraphicsManager.getRenderableMeshVertexDeclaration());

	Renderer.setVertexBuffer(renderableMesh.getVertexBuffer());
	Renderer.setIndexBuffer(renderableMesh.getIndexBuffer());

	mtx worldTransform = mtx::rotateX(-PI/2.0f);

	// shadow-map pass
	if (lightViewProjTransform == mtx::identity())
	{
		Renderer.setVertexShaderConstant("worldViewProjTransform", worldTransform * viewProjTransform);
	}
	// light pass
	else
	{
		Renderer.setVertexShaderConstant("worldTransform", worldTransform);
		Renderer.setVertexShaderConstant("worldViewProjTransform", worldTransform * viewProjTransform);
		Renderer.setVertexShaderConstant("lightWorldViewProjTransform", worldTransform * lightViewProjTransform);
	}

	Renderer.drawIndexedPrimitives(PrimitiveType::TriangleList, mesh.getFacesNum(), 0, 0, mesh.getVerticesNum());
}



void run()
{
	float deltaTime = (float)Application.getLastFrameTime();



	updateCamera(deltaTime);



	vec3 lightDirection = vec3(-1.0f, -1.0f, -1.0f).getNormalized();

	mtx viewProjTransform =
		mtx::lookAtRH(camera.getEye(), camera.getAt(), camera.getUp()) *
		mtx::perspectiveFovRH(PI/3.0f, Application.getScreenAspectRatio(), 0.1f, 200.0f);

	mtx lightViewProjTransform =
		mtx::lookAtRH(-100.0f*lightDirection, vec3(), vec3(0.0f, 1.0f, 0.0f)) *
		mtx::orthoRH(100.0f, 100.0f, 10.0f, 500.0f);



	static int shadowMapPS_techniqueID = 0;
	static int lightPS_techniqueID = 0;
	static CRenderTarget *currentShadowMapRT1 = &shadowMap64RT1;
	static CRenderTarget *currentShadowMapRT2 = &shadowMap64RT2;
	static bool blurring = false;
	static float VSM_tailCutOff = 0.05f;
	static float ESM_k = 90.0f;

	if (Application.isKeyPressed(SDLK_F1))
	{
		shadowMapPS_techniqueID = 0;
		lightPS_techniqueID = 0;
		currentShadowMapRT1 = &shadowMap32RT1;
		currentShadowMapRT2 = &shadowMap32RT2;
	}
	else if (Application.isKeyPressed(SDLK_F2))
	{
		shadowMapPS_techniqueID = 1;
		lightPS_techniqueID = 1;
		currentShadowMapRT1 = &shadowMap64RT1;
		currentShadowMapRT2 = &shadowMap64RT2;
	}
	else if (Application.isKeyPressed(SDLK_F3))
	{
		shadowMapPS_techniqueID = 1;
		lightPS_techniqueID = 2;
		currentShadowMapRT1 = &shadowMap64RT1;
		currentShadowMapRT2 = &shadowMap64RT2;
	}
	else if (Application.isKeyPressed(SDLK_F4))
	{
		shadowMapPS_techniqueID = 2;
		lightPS_techniqueID = 3;
		currentShadowMapRT1 = &shadowMap32RT1;
		currentShadowMapRT2 = &shadowMap32RT2;
	}
	else if (Application.isKeyPressed(SDLK_F5))
	{	
		shadowMapPS_techniqueID = 3;
		lightPS_techniqueID = 4;
		currentShadowMapRT1 = &shadowMap64RT1;
		currentShadowMapRT2 = &shadowMap64RT2;
	}

	if (Application.isKeyPressed(SDLK_q))
		blurring = false;
	else if (Application.isKeyPressed(SDLK_e))
		blurring = true;

	if (Application.isKeyPressed(SDLK_r))
		VSM_tailCutOff += 0.0001f * deltaTime;
	else if (Application.isKeyPressed(SDLK_f))
		VSM_tailCutOff -= 0.0001f * deltaTime;

	if (Application.isKeyPressed(SDLK_t))
		ESM_k += 0.02f * deltaTime;
	else if (Application.isKeyPressed(SDLK_g))
		ESM_k -= 0.02f * deltaTime;



	ostringstream oss;

	if (lightPS_techniqueID == 0)
		oss << "SSM";
	else if (lightPS_techniqueID == 1)
		oss << "VSM, tail cut off = " << VSM_tailCutOff;
	else if (lightPS_techniqueID == 2)
		oss << "VSM applied to shadow boundaries, tail cut off = " << VSM_tailCutOff;
	else if (lightPS_techniqueID == 3)
		oss << "ESM, k = " << ESM_k;
	else if (lightPS_techniqueID == 4)
		oss << "EVSM, tail cut off = " << VSM_tailCutOff << ", k = " << ESM_k;

	Application.setWindowText(oss.str());

	oss.str(string());



	Renderer.beginScene();



	Renderer.setDepthStencilSurface(shadowMapDSS);

	// shadow-map filling
	Renderer.setRenderTarget(0, *currentShadowMapRT1);
	Renderer.clear(true, true, false, vec3(1.0f, 1.0f, 1.0f));
	{
		Renderer.setVertexShader(shadowMapVS);
		Renderer.setPixelShader(shadowMapPS[shadowMapPS_techniqueID]);
		{
			Renderer.setPixelShaderConstant("ESM_k", ESM_k);
		}
		Renderer.setTextureOff(0);
		Renderer.setTextureOff(1);
		Renderer.setTextureOff(2);
		renderSceneGeometry(lightViewProjTransform, mtx::identity());
	}

	// blurring the shadow-map
	if (blurring)
	{
		Renderer.setVertexDeclaration(screenVD);

		Renderer.setRenderTarget(0, *currentShadowMapRT2);
		Renderer.clear(true, true, false, vec3(1.0f, 1.0f, 1.0f));
		{
			Renderer.setVertexShader(screenQuadVS);
			Renderer.setPixelShader(shadowMapHorizontalBlurPS);
			{
				Renderer.setPixelShaderConstant("texelWidth", 1.0f/(float)shadowMapSize);
			}
			Renderer.setTexture(0, "shadowMapSampler", *currentShadowMapRT1);

			Renderer.setVertexBuffer(screenVB);
			Renderer.drawPrimitives(PrimitiveType::TriangleStrip, 2, 0);
		}
		Renderer.setRenderTarget(0, *currentShadowMapRT1);
		Renderer.clear(true, true, false, vec3(1.0f, 1.0f, 1.0f));
		{
			Renderer.setVertexShader(screenQuadVS);
			Renderer.setPixelShader(shadowMapVerticalBlurPS);
			{
				Renderer.setPixelShaderConstant("texelHeight", 1.0f/(float)shadowMapSize);
			}
			Renderer.setTexture(0, "shadowMapSampler", *currentShadowMapRT2);

			Renderer.setVertexBuffer(screenVB);
			Renderer.drawPrimitives(PrimitiveType::TriangleStrip, 2, 0);
		}
	}

	Renderer.setDepthStencilSurfaceToBackBuffer();

	// final rendering
	Renderer.setRenderTargetToBackBuffer(0);
	Renderer.clear(true, true, false, vec3(0.5f, 0.5f, 0.5f));
	{
		Renderer.setVertexShader(lightVS);
		Renderer.setPixelShader(lightPS[lightPS_techniqueID]);
		{
			Renderer.setPixelShaderConstant("shadowMapSize", (float)shadowMapSize);
			Renderer.setPixelShaderConstant("VSM_tailCutOff", VSM_tailCutOff);
			Renderer.setPixelShaderConstant("ESM_k", ESM_k);
			Renderer.setPixelShaderConstant("lightDirection", lightDirection, 0.0f);
		}
		Renderer.setTexture(0, "shadowMapSampler_point",  *currentShadowMapRT1);
		Renderer.setTexture(1, "shadowMapSampler_linear", *currentShadowMapRT1);
		Renderer.setTexture(2, "diffuseMapSampler", texture);
		renderSceneGeometry(viewProjTransform, lightViewProjTransform);
	}



	Renderer.endScene();



	if (Application.isKeyPressed(SDLK_ESCAPE))
		Application.breakRunFunction();
}



#ifdef WIN32
	#undef main
#endif

int main(int argc, char *argv[])
{
	#ifdef WIN32
		#if defined(DEBUG) | defined(_DEBUG)
			_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
		#endif
	#endif

	if (argc == 2)
		shadowMapSize = atoi(argv[1]);

	bool createOpenGLContext = false;
	#ifdef BLOSSOM_RENDERER_OGL
		createOpenGLContext = true;
	#endif

	Logger.open("log_BlossomEngineApplication.txt");
	Application.create(800, 600, 0, false, false, createOpenGLContext);
	Application.setMouseMotionFunction(mouseMotionFunction);
	GraphicsManager.create(Application.getScreenWidth(), Application.getScreenHeight(), Application.isFullScreen(), Application.isVSync());

	init();
	Application.run(run);

	GraphicsManager.destroy();
	Application.destroy();
	Logger.close();

	return 0;
}
