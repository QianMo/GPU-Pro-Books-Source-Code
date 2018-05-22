
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "App.h"

BaseApp *app = new App();

void App::moveCamera(const float3 &dir)
{
	float3 newPos = camPos + dir * (speed * frameTime);

	float3 point;
	const BTri *tri;
	if (m_BSP.intersects(camPos, newPos, &point, &tri))
	{
		newPos = point + tri->plane.xyz();
	}
	m_BSP.pushSphere(newPos, 50);

	camPos = newPos;
}

void App::resetCamera()
{
	camPos = vec3(1992, 30, -432);
	wx = 0.078f;
	wy = 0.62f;
}

bool App::onKey(const uint key, const bool pressed)
{
	if (D3D10App::onKey(key, pressed)) return true;

	if (pressed)
	{
		if (key == KEY_F5)
			m_UseSDAA->setChecked(!m_UseSDAA->isChecked());
		else if (key == KEY_F6)
			m_UseSDPreZ->setChecked(!m_UseSDPreZ->isChecked());
	}

	return false;
}

void App::onSize(const int w, const int h)
{
	D3D10App::onSize(w, h);

	if (renderer)
	{
		// Make sure render targets are the size of the window
		renderer->resizeRenderTarget(m_BaseRT,    w, h, 1, 1, 1);
		renderer->resizeRenderTarget(m_NormalRT,  w, h, 1, 1, 1);
		renderer->resizeRenderTarget(m_DepthRT,   w, h, 1, 1, 1);
		renderer->resizeRenderTarget(m_SDDepthRT, w, h, 1, 1, 1);
		renderer->resizeRenderTarget(m_SDColorRT, w, h, 1, 1, 1);
		renderer->resizeRenderTarget(m_ResultRT,  w, h, 1, 1, 1);
	}
}

bool App::init()
{
	// No framework-created depth buffer
	depthBits = 0;

	// Load map
	m_Map = new Model();
	if (!m_Map->loadObj("../Models/Corridor2/MapFixed.obj")) return false;
	m_Map->scale(0, float3(1, 1, -1));

	// Create BSP for collision detection
	uint nIndices = m_Map->getIndexCount();
	float3 *src = (float3 *) m_Map->getStream(0).vertices;
	uint *inds = m_Map->getStream(0).indices;

	for (uint i = 0; i < nIndices; i += 3)
	{
		const float3 &v0 = src[inds[i]];
		const float3 &v1 = src[inds[i + 1]];
		const float3 &v2 = src[inds[i + 2]];

		m_BSP.addTriangle(v0, v1, v2);
	}
	m_BSP.build();

	m_Map->computeTangentSpace(true);

	//m_Map->cleanUp();

	// Create light sphere for deferred shading
	m_Sphere = new Model();
	m_Sphere->createSphere(3);

	// Initialize all lights
	m_Lights[ 0].position = float3( 576, 96,    0); m_Lights[ 0].radius = 640.0f;
	m_Lights[ 1].position = float3( 0,   96,  576); m_Lights[ 1].radius = 640.0f;
	m_Lights[ 2].position = float3(-576, 96,    0); m_Lights[ 2].radius = 640.0f;
	m_Lights[ 3].position = float3( 0,   96, -576); m_Lights[ 3].radius = 640.0f;
	m_Lights[ 4].position = float3(1792, 96,  320); m_Lights[ 4].radius = 550.0f;
	m_Lights[ 5].position = float3(1792, 96, -320); m_Lights[ 5].radius = 550.0f;
	m_Lights[ 6].position = float3(-192, 96, 1792); m_Lights[ 6].radius = 550.0f;
	m_Lights[ 7].position = float3(-832, 96, 1792); m_Lights[ 7].radius = 550.0f;
	m_Lights[ 8].position = float3(1280, 32,  192); m_Lights[ 8].radius = 450.0f;
	m_Lights[ 9].position = float3(1280, 32, -192); m_Lights[ 9].radius = 450.0f;
	m_Lights[10].position = float3(-320, 32, 1280); m_Lights[10].radius = 450.0f;
	m_Lights[11].position = float3(-704, 32, 1280); m_Lights[11].radius = 450.0f;
	m_Lights[12].position = float3( 960, 32,  640); m_Lights[12].radius = 450.0f;
	m_Lights[13].position = float3( 960, 32, -640); m_Lights[13].radius = 450.0f;
	m_Lights[14].position = float3( 640, 32, -960); m_Lights[14].radius = 450.0f;
	m_Lights[15].position = float3(-640, 32, -960); m_Lights[15].radius = 450.0f;
	m_Lights[16].position = float3(-960, 32,  640); m_Lights[16].radius = 450.0f;
	m_Lights[17].position = float3(-960, 32, -640); m_Lights[17].radius = 450.0f;
	m_Lights[18].position = float3( 640, 32,  960); m_Lights[18].radius = 450.0f;

	// Init GUI components
	int tab = configDialog->addTab("SDAA");
	configDialog->addWidget(tab, m_UseSDAA = new CheckBox(0, 0, 150, 36, "Use SDAA", true));
	configDialog->addWidget(tab, m_UseSDPreZ = new CheckBox(0, 40, 350, 36, "Use second-depth as pre-Z", true));

	return true;
}

void App::exit()
{
	delete m_Sphere;
	delete m_Map;
}

bool App::initAPI()
{
	// Override the user's MSAA settings and disable the control
	antiAlias->setEnabled(false);
	return D3D10App::initAPI(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_UNKNOWN, 1, NO_SETTING_CHANGE);
}

void App::exitAPI()
{
	D3D10App::exitAPI();
}

bool App::load()
{
	// Shaders
	if ((m_DepthFill   = renderer->addShader("DepthFill.shd"  )) == SHADER_NONE) return false;
	if ((m_FillBuffers = renderer->addShader("FillBuffers.shd")) == SHADER_NONE) return false;
	if ((m_Ambient     = renderer->addShader("Ambient.shd"    )) == SHADER_NONE) return false;
	if ((m_Lighting    = renderer->addShader("Lighting.shd"   )) == SHADER_NONE) return false;
#ifdef USE_D3D10_1
	if ((m_AntiAlias   = renderer->addShader("SDAA.shd", "#define DX10_1\n")) == SHADER_NONE) return false;
#else
	if ((m_AntiAlias   = renderer->addShader("SDAA.shd")) == SHADER_NONE) return false;
#endif

	// Samplerstates
	if ((m_BaseFilter = renderer->addSamplerState(TRILINEAR_ANISO, WRAP, WRAP, WRAP)) == SS_NONE) return false;
	if ((m_PointClamp = renderer->addSamplerState(NEAREST, CLAMP, CLAMP, CLAMP)) == SS_NONE) return false;

	// Main render targets
	if ((m_BaseRT    = renderer->addRenderTarget(width, height, 1, 1, 1, FORMAT_RGBA8,  1, SS_NONE, SRGB        )) == TEXTURE_NONE) return false;
	if ((m_NormalRT  = renderer->addRenderTarget(width, height, 1, 1, 1, FORMAT_RGBA8S, 1, SS_NONE              )) == TEXTURE_NONE) return false;
	if ((m_DepthRT   = renderer->addRenderDepth (width, height, 1,       FORMAT_D32F,   1, SS_NONE, SAMPLE_DEPTH)) == TEXTURE_NONE) return false;
	if ((m_SDDepthRT = renderer->addRenderDepth (width, height, 1,       FORMAT_D32F,   1, SS_NONE, SAMPLE_DEPTH)) == TEXTURE_NONE) return false;
	if ((m_SDColorRT = renderer->addRenderTarget(width, height, 1, 1, 1, FORMAT_R32F,   1, SS_NONE              )) == TEXTURE_NONE) return false;
	if ((m_ResultRT  = renderer->addRenderTarget(width, height, 1, 1, 1, FORMAT_RGBA8,  1, SS_NONE, SRGB        )) == TEXTURE_NONE) return false;

	// Textures
	if ((m_BaseTex[0] = renderer->addTexture  ("../Textures/wood.dds",                    true, m_BaseFilter, SRGB)) == TEXTURE_NONE) return false;
	if ((m_BumpTex[0] = renderer->addNormalMap("../Textures/woodBump.dds", FORMAT_RGBA8S, true, m_BaseFilter      )) == TEXTURE_NONE) return false;

	if ((m_BaseTex[1] = renderer->addTexture  ("../Textures/Tx_imp_wall_01_small.dds",              true, m_BaseFilter, SRGB)) == TEXTURE_NONE) return false;
	if ((m_BumpTex[1] = renderer->addNormalMap("../Textures/Tx_imp_wall_01Bump.dds", FORMAT_RGBA8S, true, m_BaseFilter      )) == TEXTURE_NONE) return false;

	if ((m_BaseTex[2] = renderer->addTexture  ("../Textures/floor_wood_3.dds",                    true, m_BaseFilter, SRGB)) == TEXTURE_NONE) return false;
	if ((m_BumpTex[2] = renderer->addNormalMap("../Textures/floor_wood_3Bump.dds", FORMAT_RGBA8S, true, m_BaseFilter      )) == TEXTURE_NONE) return false;

	if ((m_BaseTex[3] = renderer->addTexture  ("../Textures/light2.dds",                    true, m_BaseFilter, SRGB)) == TEXTURE_NONE) return false;
	if ((m_BumpTex[3] = renderer->addNormalMap("../Textures/light2Bump.dds", FORMAT_RGBA8S, true, m_BaseFilter      )) == TEXTURE_NONE) return false;

	if ((m_BaseTex[4] = renderer->addTexture  ("../Textures/floor_wood_4.dds",                    true, m_BaseFilter, SRGB)) == TEXTURE_NONE) return false;
	if ((m_BumpTex[4] = renderer->addNormalMap("../Textures/floor_wood_4Bump.dds", FORMAT_RGBA8S, true, m_BaseFilter      )) == TEXTURE_NONE) return false;

	// Blendstates
	if ((m_BlendAdd = renderer->addBlendState(ONE, ONE)) == BS_NONE) return false;

	// Depth states - use reversed depth (1 to 0) to improve precision
	if ((m_DepthTestGEqual = renderer->addDepthState(true, true, GEQUAL)) == DS_NONE) return false;

	// Upload map to vertex/index buffer
	if (!m_Map->makeDrawable(renderer, true, m_FillBuffers)) return false;
	if (!m_Sphere->makeDrawable(renderer, true, m_Lighting)) return false;

	return true;
}

void App::unload()
{

}

void App::drawFrame()
{
	const float near_plane = 20.0f;
	const float far_plane = 4000.0f;

	// Reversed depth
	float4x4 projection = toD3DProjection(perspectiveMatrixY(1.2f, width, height, far_plane, near_plane));
	float4x4 view = rotateXY(-wx, -wy);
	view.translate(-camPos);
	float4x4 viewProj = projection * view;
	// Pre-scale-bias the matrix so we can use the screen position directly
	float4x4 viewProjInv = (!viewProj) * (translate(-1.0f, 1.0f, 0.0f) * scale(2.0f, -2.0f, 1.0f));


	Direct3D10Renderer *d3d10_renderer = (Direct3D10Renderer *) renderer;


	if (m_UseSDAA->isChecked() && !m_UseSDPreZ->isChecked())
	{
		/*
			Separate second-depth pass.
		*/
		renderer->changeRenderTargets(NULL, 0, m_SDDepthRT);
		{
			renderer->clear(false, true, false, NULL, 0.0f);

			renderer->reset();
			renderer->setRasterizerState(cullFront);
			renderer->setDepthState(m_DepthTestGEqual);
			renderer->setShader(m_DepthFill);
			renderer->setShaderConstant4x4f("ViewProj", viewProj);
			renderer->apply();

			m_Map->draw(renderer);
		}
	}


	TextureID bufferRTs[] = { m_BaseRT, m_NormalRT };
	renderer->changeRenderTargets(bufferRTs, elementsOf(bufferRTs), m_DepthRT);
	{
		renderer->clear(false, true, false, NULL, 0.0f);

		/*
			Pre-z pass.

			Here we either do a regular pre-z pass, or render the second-depth value, which will also work as a pre-Z pass.
			Depending on scene layout second-depth may be faster or slower than regular pre-z pass. In this demo second-depth is faster.
			All backfacing triangles are guaranteed to cover some frontbacking surface behind it, but not all frontfacing triangles
			cover any other frontfacing surface, for instance walls in front of empty space. Hence backfacing are fully efficient as
			occluders whereas frontfacing will have some amount of waste.
		*/
		if (!m_UseSDAA->isChecked() || m_UseSDPreZ->isChecked())
		{
			renderer->reset();
			renderer->setRasterizerState(m_UseSDPreZ->isChecked()? cullFront : cullBack);
			renderer->setDepthState(m_DepthTestGEqual);
			renderer->setShader(m_DepthFill);
			renderer->setShaderConstant4x4f("ViewProj", viewProj);
			renderer->apply();

			m_Map->draw(renderer);


			if (m_UseSDAA->isChecked() && m_UseSDPreZ->isChecked())
			{
				// Copy depth to second-depth texture
				device->CopyResource(d3d10_renderer->getResource(m_SDColorRT), d3d10_renderer->getResource(m_DepthRT));
			}
		}


		/*
			Main scene pass.
			This is where the buffers are filled for the later deferred passes.
		*/
		renderer->reset();
		renderer->setRasterizerState(cullBack);
		renderer->setDepthState(m_DepthTestGEqual);
		renderer->setShader(m_FillBuffers);
		renderer->setShaderConstant4x4f("ViewProj", viewProj);
		renderer->setSamplerState("Filter", m_BaseFilter);
		renderer->apply();

		uint count = m_Map->getBatchCount();
		for (uint i = 0; i < count; i++)
		{
			renderer->setTexture("Base", m_BaseTex[i]);
			renderer->setTexture("Bump", m_BumpTex[i]);
			renderer->applyTextures();

			m_Map->drawBatch(renderer, i);
		}
	}


	TextureID target = m_UseSDAA->isChecked()? m_ResultRT : FB_COLOR;

	renderer->changeRenderTarget(target, TEXTURE_NONE);

	{
		/*
			Deferred ambient pass.
		*/
		renderer->reset();
		renderer->setRasterizerState(cullNone);
		renderer->setDepthState(noDepthTest);
		renderer->setShader(m_Ambient);
		renderer->setTexture("Base", m_BaseRT);
		renderer->setSamplerState("Filter", m_PointClamp);
		renderer->apply();


		device->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		device->Draw(3, 0);


		/*
			Deferred lighting pass.
		*/
		renderer->reset();
		renderer->setDepthState(noDepthTest);
		renderer->setShader(m_Lighting);
		renderer->setRasterizerState(cullFront);
		renderer->setBlendState(m_BlendAdd);
		renderer->setShaderConstant4x4f("ViewProj", viewProj);
		renderer->setShaderConstant4x4f("ViewProjInv", viewProjInv * scale(1.0f / width, 1.0f / height, 1.0f));
		renderer->setShaderConstant3f("CamPos", camPos);
		renderer->setTexture("Base", m_BaseRT);
		renderer->setTexture("Normal", m_NormalRT);
		renderer->setTexture("Depth", m_DepthRT);
		renderer->apply();

		float2 zw = projection.rows[2].zw();
		for (uint i = 0; i < LIGHT_COUNT; i++)
		{
			float3 lightPos = m_Lights[i].position;
			float radius = m_Lights[i].radius;
			float invRadius = 1.0f / radius;

			// Compute z-bounds
			float4 lPos = view * float4(lightPos, 1.0f);
			float z1 = lPos.z + radius;

			if (z1 > near_plane)
			{
				float z0 = max(lPos.z - radius, near_plane);

				float2 zBounds;
				zBounds.y = saturate(zw.x + zw.y / z0);
				zBounds.x = saturate(zw.x + zw.y / z1);

				renderer->setShaderConstant3f("LightPos", lightPos);
				renderer->setShaderConstant1f("Radius", radius);
				renderer->setShaderConstant1f("InvRadius", invRadius);
				renderer->setShaderConstant2f("ZBounds", zBounds);
				renderer->applyConstants();

				m_Sphere->draw(renderer);
			}
		}
	}


	/*
		Second Depth Anti-Aliasing
	*/
	if (m_UseSDAA->isChecked())
	{
		renderer->changeToMainFramebuffer();

		// SDAA resolve pass
		renderer->reset();
		renderer->setShader(m_AntiAlias);
		renderer->setRasterizerState(cullNone);
		renderer->setShaderConstant2f("PixelSize", float2(1.0f / width, 1.0f / height));
		renderer->setTexture("BackBuffer", m_ResultRT);
		renderer->setTexture("DepthBuffer", m_DepthRT);
		renderer->setTexture("SecondDepthBuffer", m_UseSDPreZ->isChecked()? m_SDColorRT : m_SDDepthRT);
		renderer->setSamplerState("Linear", linearClamp);
		renderer->setSamplerState("Point", m_PointClamp);
		renderer->apply();

		renderer->drawArrays(PRIM_TRIANGLES, 0, 3);
	}
}
