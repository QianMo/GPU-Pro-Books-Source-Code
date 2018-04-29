/*!****************************************************************************
 @File          Page.cpp

 @Title         EBook Demo

 @Copyright     Copyright 2010 by Imagination Technologies Limited.

 @Platform      Independent

 @Description   

******************************************************************************/

#include "OGLES2Tools.h"

#include "Page.h"
#include "Resource.h"

static const bool c_bRenderOnDemand = true;
static const unsigned int c_uiNumSwapBuffers = 2;

// Some constants
static const float c_fMinDrag = 0.15f;

static const float c_fMaxStrength = 3.0f;

static const float c_fFoldingSpeed = 2.6f;

#ifdef __APPLE__
static const float c_fPointerDragMult = 0.6f;
#else
static const float c_fPointerDragMult = 0.9f;
#endif

static const float c_fPointerInertiaMult = 0.04f;
static const float c_fInertiaTimeNormalization = 0.03f;

static const float c_fInputBorderWidth = 0.2f;
static const float c_fInputBorderHeight = 0.2f;

static const float c_fMinRadius = 0.001f;
static const float c_fAnglePageTurnThreshold = 0.015f / PVRT_PI;

static const float c_fZOffset = 0.01f;

// Defines for shaders
static const char *c_aszEffectTexture[] = { "ENABLE_FOLDING", "HAS_TEXTURE" };
static const char *c_aszEffectAA[] = { "ENABLE_FOLDING", "HAS_TEXTURE", "FOLD_GRADIENT" };
static const char *c_aszEffectIntBorder[] = { "ENABLE_FOLDING", "HAS_TEXTURE", "EDGE_GRADIENT", "EDGE_GRADIENT_INT" };
static const char *c_aszEffectIntBorderAA[] = { "ENABLE_FOLDING", "HAS_TEXTURE", "EDGE_GRADIENT", "EDGE_GRADIENT_INT", "FOLD_GRADIENT" };
static const char *c_aszEffectExtBorder[] = { "ENABLE_FOLDING", "EDGE_GRADIENT", "EDGE_GRADIENT_EXT" };
static const char *c_aszEffectWireframe[] = { "ENABLE_FOLDING", "HAS_TEXTURE", "ENABLE_WIREFRAME" };
static const char *c_aszEffectNoFolding[] = { "HAS_TEXTURE" };
static const char *c_aszEffectNoFoldingIntBorder[] = { "HAS_TEXTURE", "EDGE_GRADIENT", "EDGE_GRADIENT_INT" };
//static const char *c_aszEffectNoFoldingIntBorderAA[] = { "HAS_TEXTURE", "EDGE_GRADIENT", "EDGE_GRADIENT_INT", "FOLD_GRADIENT" };
static const char *c_aszEffectWireframeNoFolding[] = { "HAS_TEXTURE", "ENABLE_WIREFRAME" };

static const char **c_aszEffectDefines[Page::NumShaders] = {

	c_aszEffectTexture,
	c_aszEffectAA,
	c_aszEffectIntBorder,
	c_aszEffectIntBorderAA,
	c_aszEffectExtBorder,
	c_aszEffectWireframe,
	c_aszEffectNoFolding,
	c_aszEffectNoFoldingIntBorder,
	//c_aszEffectNoFoldingIntBorderAA,
	c_aszEffectWireframeNoFolding
};
// Number of defines for shaders
static const int c_aiEffectDefinesNum[Page::NumShaders] = {
	2,
	3,
	4,
	5,
	3,
	3,
	1,
	3,
};


/*!****************************************************************************
 Page class implementation
******************************************************************************/

Page::Page() :
	uiFrontTex(~0), uiBackTex(~0), uiCurrTex(~0),
	iPOD(~0),

	vScale(1.0f, 1.0f),
	vTranslation(0.0f, 0.0f),

	vOrigin(0.0f, 0.0f),
	vEnd(0.0f, 0.0f),
	
	bValidInput(false),

	vDirection(1.0f, 0.0f),
	vTangent(0.0f, -1.0f),
	vPoint(0.0f, 0.0f),

	bRender(false),
	pParent(false),
	vDir2(0.0f, 0.0f),
	fZOffset(0.0f),


	fAngle(0.0f), fCorrAngle(0.0f),
	fFold(0.0f), fCorrFold(0.0f),
	fRadius(0.1f), fCorrRadius(0.1f),
	
	eState(StateIdle)
{
	UpdateDirection(fAngle);
}




bool Page::Init(Book *parent, float zOffset)
{
	pParent = parent;
	fZOffset = zOffset;

	// Template method (reads values from virtuals)
	vScale = ScaleVector();
	vTranslation = TranslationVector();

	if (!LoadPOD(scene, GetPODName()))
		return false;

	// The POD file is composed by four meshes, namely:
	// "PageInt", which represents the internal area of the page
	// "IntBorder", which represents the internal border of the page
	// "ExtBorder2", which represents the external border, used to render the
	// shadow/semi-transparent effect
	// "PageIntFlat" is a non-tessellated quad that covers the internal area
	// of the page and is used for rendering when the folding value = 0
	piMeshIndex[0] = FindNodeIndex(scene, "PageInt");
	piMeshIndex[1] = FindNodeIndex(scene, "IntBorder");
	piMeshIndex[2] = FindNodeIndex(scene, "ExtBorder2");
	piMeshIndex[3] = FindNodeIndex(scene, "PageIntFlat");

	LoadVbos(scene);

	return true;
}


bool Page::Input(float t, float dt, const TouchState *pTouchState)
{
	PVRTVec2 dir;
	float angle = 0.0f;
	bool updateDir = false;
	// reset to "page does not need to be rendered" each frame
	bRender = false;

	// OnPressed: store origin and end position
	if (pTouchState->IsPressed())
	{
		bValidInput = InputCondition(pTouchState) && fFold == 0.0f;
		if (bValidInput)
		{
			vEnd.x = vOrigin.x = pTouchState->GetPositionX();
			vEnd.x = vOrigin.y = pTouchState->GetPositionY();
		}
		// If gesture not recognized, unfold automatically
		else
		{
			eState = StateUnfolding;
		}
		// either way, redraw page
		bRender = true;
	}
	// Update durection/angle
	if ((pTouchState->IsDragging() || pTouchState->IsReleased()) && bValidInput)
	{
		vEnd.x = pTouchState->GetPositionX();
		vEnd.y = pTouchState->GetPositionY();

		float len = (vEnd - vOrigin).length();
		if (len > c_fMinDrag)
		{
			fFold = c_fPointerDragMult * (len - c_fMinDrag);
	
			dir = (vEnd - vOrigin).normalize();
			angle = atan2(dir.y, dir.x);
			// touch position changed: redraw page
			bRender = true;
			updateDir = true;
		}
	}
	// input gesture accepted to drag page
	if (bValidInput)
	{
		PVRTVec2 inertia(pTouchState->GetInertiaX(), pTouchState->GetInertiaY());
		// Normalize to delta time (heuristic to prevent too high values for high
		// framerates)
		inertia *= dt / c_fInertiaTimeNormalization;
		// mult is in the [-|inertia|, |inertia|] range and defines how the fold
		// value is altered. the fold value increases if mult is in the same
		// direction of vDir2 (positive), and decreases otherwise
		float mult = inertia.dot(vDir2);
		// Inertia is calculated as an exponential, ignore it for very small values
		// (this causes less render on demand)
		if (fabs(mult) > 0.001)
		{
			bRender = true;
			fFold += mult * c_fPointerInertiaMult;
		}
	}
	
	if (fFold > c_fMaxStrength)
		fFold = c_fMaxStrength;

	// handle unfolding
	if (eState == StateUnfolding)
	{
		// this automatically changes the fold value, therefore the page needs to be rendered
		bRender = true;

		if ((fFold -= c_fFoldingSpeed * dt) <= 0.0f)
		{
			fFold = 0.0f;
			eState = StateIdle;
		}
	}
	if (updateDir)
		UpdateState(dir, angle);

	UpdateDirection(fAngle);

	fCorrRadius = fRadius;
	fCorrAngle = fAngle;
	fCorrFold = fFold;

	// Derived classes can change the state here, for example to prevent
	// page tearing
	PageTurn();

	
	bool ret = true;
	// Page completely folded
	if (PageTurnComplete())
	{
		fCorrFold = fFold = 0.0f;
		bValidInput = false;
		ret = false;
	}
	// Update values that will be passed as uniforms
	// Corner on screen to fold from (can be either [-1,-1], [-1,1], [1,-1], [1,1]) (Pre-multiply by aspect ratio)
	PVRTVec2 corner;
	UpdateCorner(corner);
	// Add the z component to the direction
	vDirection = PVRTVec2(vDir2.x, vDir2.y);
	// Pick the vector orthogonal to the direction on the plane. This vector defines the direction of the bending axis
	vTangent = PVRTVec2(-vDir2.y, vDir2.x);
	// Calculate a point on the bending axis, and shift it to the point of minimum distance with respect to the
	// input vertex. Such point will be orthonormal to vTan
	vPoint = corner + vDirection * fCorrFold;
	
	return ret;
}

void Page::UpdateState(const PVRTVec2 &dir, const float angle)
{
	vDir2 = dir;
	fAngle = angle;
}


void Page::UpdateCorner(PVRTVec2 &corner)
{
	corner.x = (vDir2.x > 0.0f) ? -vScale.x : vScale.x;
	corner.y = (vDir2.y > 0.0f) ? -vScale.y : vScale.y;	
	
	corner.x *= pParent->AspectRatio();
}

float Page::ClampRadius()
{
	// Radius cannot be smaller than 0
	if (fCorrRadius < 0.0f)
	{
		float temp = -fCorrRadius + c_fMinRadius;
		fCorrRadius = c_fMinRadius;
		return temp;
	}
	return 0.0f;
}

void Page::UpdateDirection(float angle)
{
	vDir2.x = cos(angle);
	vDir2.y = sin(angle);
}

void Page::SetupShader(const Shader *shader) const
{
	glUseProgram(shader->id);

	glUniform1f(shader->RadiusLoc, fCorrRadius);

	// Precalculated values
	glUniform2fv(shader->DirectionLoc, 1, &vDirection.x);
	glUniform2fv(shader->TangentLoc, 1, &vTangent.x);
	glUniform2fv(shader->PointLoc, 1, &vPoint.x);

	glUniform1f(shader->AspectRatioLoc, pParent->AspectRatio());

	glUniform1f(shader->ZOffsetLoc, fZOffset);	

	glUniform1i(shader->FlipTextureLoc, uiCurrTex == uiBackTex ? 1 : 0);

	glUniform2fv(shader->InputScaleLoc, 1, &vScale.x);
	glUniform2fv(shader->InputTranslationLoc, 1, &vTranslation.x);

	glUniform1i(shader->RotateLoc, pParent->Rotated());

}

void Page::SetTextures(GLuint front, GLuint back)
{
	uiFrontTex = front;
	uiBackTex = back;
}

void Page::Render(PageMode pageMode) const
{
	glCullFace(FrontFace());
	// FIXME: For some reason, when using Print3D the textures are mapped incorrectly.
	// The code above is a workaround.
#ifdef DEBUG_MODE
	glBindTexture(GL_TEXTURE_2D, uiCurrTex = uiFrontTex);
#else
	glBindTexture(GL_TEXTURE_2D, uiCurrTex = uiBackTex);
#endif
	RenderSide(pageMode);

	glCullFace(BackFace());
#ifdef DEBUG_MODE
	glBindTexture(GL_TEXTURE_2D, uiCurrTex = uiBackTex);
#else
	glBindTexture(GL_TEXTURE_2D, uiCurrTex = uiFrontTex);
#endif
	RenderSide(pageMode);
}



void Page::RenderSide(PageMode pageMode) const
{
	/* GL rendering */
	if ((pageMode & Page::PageInside))
	{
		// shader1 is the program used to render the internal area
		// shader2 is the program used to render the internal border
		const Shader *shader1, *shader2;
		// Index 3 corresponds to the simple non-tessellated quad
		int index;
		if (fFold == 0.0f)
		{
			index = 3;
			if (!pParent->Wireframe())
			{
				shader1 = GetShader(ShaderNoFolding);
				shader2 = GetShader(ShaderNoFoldingIntBorder);
			}
			else
				shader1 = shader2 = GetShader(ShaderNoFoldingWireframe);
		}
		else
		{
			index = 0;
			// Render Page
			if (!pParent->Wireframe())
			{
				shader1 = pParent->AAEnabled() ? GetShader(ShaderFoldingAA) : GetShader(ShaderFolding);
				shader2 = pParent->AAEnabled() ? GetShader(ShaderFoldingIntBorderAA): GetShader(ShaderFoldingIntBorder);
			}
			else
				shader1 = shader2 = GetShader(ShaderFoldingWireframe);
		}
		// Render
		SetupShader(shader1);

		DrawMesh(index, pParent->Wireframe());

		// Render Page Border
		SetupShader(shader2);

		// Draw mesh
		DrawMesh(1, pParent->Wireframe());

	}
	if ((pageMode & Page::PageOutside) && !pParent->Wireframe())
	{
		// Render Page Border
		SetupShader(GetShader(ShaderFoldingExtBorder));

		// Draw mesh
		DrawMesh(2, pParent->Wireframe());
	}
}

void Page::RenderDebug() const
{
	float v[] = { vOrigin.x, vOrigin.y, 0.0f, vEnd.x, vEnd.y, 0.0f };

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, v);
	glDrawArrays(GL_LINES, 0, 2);
	glDisableVertexAttribArray(0);
}

const Page::Shader *Page::GetShader(PageShaders shader) const
{
	return pParent->GetShader(shader);
}



bool Page::LoadVbos(CPVRTModelPOD &scene)
{
	if(!scene.pMesh[0].pInterleaved)
	{
		return false;
	}

	/*
		Load vertex data of all meshes in the scene into VBOs

		The meshes have been exported with the "Interleave Vectors" option,
		so all data is interleaved in the buffer at pMesh->pInterleaved.
		Interleaving data improves the memory access pattern and cache efficiency,
		thus it can be read faster by the hardware.
	*/
	glGenBuffers(4, m_puiVbo);
	for (unsigned int i = 0; i < 4; ++i)
	{
		// Load vertex data into buffer object
		SPODMesh& Mesh = scene.pMesh[piMeshIndex[i]];
		unsigned int uiSize = Mesh.nNumVertex * Mesh.sVertex.nStride;
		glBindBuffer(GL_ARRAY_BUFFER, m_puiVbo[i]);
		glBufferData(GL_ARRAY_BUFFER, uiSize, Mesh.pInterleaved, GL_STATIC_DRAW);

		// Load index data into buffer object if available
		m_puiIndexVbo[i] = 0;
		if (Mesh.sFaces.pData)
		{
			glGenBuffers(1, &m_puiIndexVbo[i]);
			uiSize = PVRTModelPODCountIndices(Mesh) * sizeof(GLshort);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_puiIndexVbo[i]);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, uiSize, Mesh.sFaces.pData, GL_STATIC_DRAW);
		}
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	return true;
}


void Page::DrawMesh(int iVBO, bool wireframe) const
{
	int i32MeshIndex = scene.pNode[piMeshIndex[iVBO]].nIdx;
	SPODMesh* pMesh = &scene.pMesh[i32MeshIndex];

	// bind the VBO for the mesh
	glBindBuffer(GL_ARRAY_BUFFER, m_puiVbo[iVBO]);
	// bind the index buffer, won't hurt if the handle is 0
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_puiIndexVbo[iVBO]);

	// Enable the vertex attribute arrays
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	// Set the vertex attribute offsets
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, pMesh->sVertex.nStride, pMesh->sVertex.pData);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, pMesh->psUVW[0].nStride, pMesh->psUVW[0].pData);

	/*
		The geometry can be exported in 4 ways:
		- Indexed Triangle list
		- Non-Indexed Triangle list
		- Indexed Triangle strips
		- Non-Indexed Triangle strips
	*/
	if(pMesh->nNumStrips == 0)
	{
		if(m_puiIndexVbo[iVBO])
		{
			// Indexed Triangle list
			glDrawElements(wireframe ? GL_LINES : GL_TRIANGLES, pMesh->nNumFaces*3, GL_UNSIGNED_SHORT, 0);
		}
		else
		{
			// Non-Indexed Triangle list
			glDrawArrays(wireframe ? GL_LINES : GL_TRIANGLES, 0, pMesh->nNumFaces*3);
		}
	}
	else
	{
		int offset = 0;

		for(int i = 0; i < (int)pMesh->nNumStrips; ++i)
		{
			if(m_puiIndexVbo[iVBO])
			{
				// Indexed Triangle strips
				glDrawElements(wireframe ? GL_LINE_STRIP : GL_TRIANGLE_STRIP, pMesh->pnStripLength[i]+2, GL_UNSIGNED_SHORT, &((GLshort*)0)[offset]);
			}
			else
			{
				// Non-Indexed Triangle strips
				glDrawArrays(wireframe ? GL_LINE_STRIP : GL_TRIANGLE_STRIP, offset, pMesh->pnStripLength[i]+2);
			}
			offset += pMesh->pnStripLength[i]+2;
		}
	}

	// Safely disable the vertex attribute arrays
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


/*!****************************************************************************
 HalfPage class implementation
******************************************************************************/

bool HalfPage::PageTurn()
{
	if (!CanFold(vDir2.x))
	{
		fCorrFold = 0.0f;
	}
	else
	{
		float aspect = pParent->AspectRatio();
		float x = fabs(fFold / vDir2.x);

		if (x > aspect)
		{
			float offsetSign = vDir2.x * vDir2.y < 0.0f
				? 1.0f : -1.0f;
			float offsetAmplitude = fFold * (x - aspect) / x;
			
			// If the folding factor exceeds what permitted without tearing
			// the page, the angle must be corrected first, and the radius next
			fCorrAngle = fAngle + offsetSign * offsetAmplitude;
			bool ret = UpdateAngleAndRadius();

			UpdateDirection(fCorrAngle);

			x = fabs(fFold / vDir2.x);
			if (x > aspect)
				fCorrFold = fFold * aspect / x;
			else
				fCorrFold = fFold;

			return ret;
		}
	}
	return false;
}

void HalfPage::UpdateState(const PVRTVec2 &dir, const float angle)
{
	if (CanFold(dir.x))
	{
		vDir2 = dir;
		fAngle = angle;
	}
	else if (fabs(dir.x) > 0.01)
	{
		vDir2.y = dir.y;
		vDir2.x = -dir.x;
		fAngle = atan2(vDir2.y, vDir2.x);
	}
}

bool HalfPage::PageTurnComplete()
{
	return fCorrRadius == c_fMinRadius;
}


/*!****************************************************************************
 LeftPage class implementation
******************************************************************************/

LeftPage::LeftPage()
{
	fRadius = 0.1f;
}

bool LeftPage::InputCondition(const TouchState *pTouchState)
{
	float x = pTouchState->GetPositionX();
	//float y = pTouchState->GetPositionY();

	return x < -1.0f + c_fInputBorderWidth;
		//|| x < 0.0f && fabs(y) > 1.0f - c_fInputBorderHeight;
}

bool LeftPage::CanFold(float xDir) const
{
	return xDir > 0.0f;
}

bool LeftPage::AngleSnap(float angle, float corrAngle) const
{
	return angle * corrAngle <= 0.0f;
}


bool LeftPage::UpdateAngleAndRadius()
{
	if (AngleSnap(fAngle, fCorrAngle))
	{
		fCorrRadius -= fabs(fCorrAngle);
		fCorrAngle = 0.0f;
		return ClampRadius();
	}
	return false;
}





/*!****************************************************************************
 RightPage class implementation
******************************************************************************/

RightPage::RightPage()
{
	// Use slightly different radiuses for left and right page to minizize
	// z-fighting (should left and right page be folded at the same time)
	fRadius = 0.101f;
}

bool RightPage::InputCondition(const TouchState *pTouchState)
{
	float x = pTouchState->GetPositionX();
	//float y = pTouchState->GetPositionY();

	return x > 1.0f - c_fInputBorderWidth;
		//|| x > 0.0f && fabs(y) > 1.0f - c_fInputBorderHeight;
}

bool RightPage::CanFold(float xDir) const
{
	return xDir < 0.0f;
}

bool RightPage::AngleSnap(float angle, float corrAngle) const
{
	return angle > 0.0f ?
		(angle - PVRT_PI) * (corrAngle - PVRT_PI) <= 0.0f :
		(angle + PVRT_PI) * (corrAngle + PVRT_PI) <= 0.0f;
}

bool RightPage::UpdateAngleAndRadius()
{
	if (AngleSnap(fAngle, fCorrAngle))
	{
		fCorrRadius -= fabs(fabs(fCorrAngle) - PVRT_PI);
		fCorrAngle = fCorrAngle > 0.0f ? PVRT_PI : -PVRT_PI;
		return ClampRadius();
	}
	return false;
}


/*!****************************************************************************
 Book class implementation
******************************************************************************/

Book::Book(PVRShell *shell) :
	pShell(shell),
	nPages(0),
	puiTexture(NULL),
	uiLeftCurPage(0),
	bWireframe(false),
	fAspectRatio(1.0f),
	bUseAA(true),
	bRenderOnDemand(c_bRenderOnDemand),
	bRenderOnDemandInit(false),
	uiNumSwapBuffers(c_uiNumSwapBuffers),
	uiFilledBuffers(0),
	bRenderThisFrame(true)
{

}

Book::~Book()
{
	delete [] puiTexture;
}


bool Book::Init(unsigned int pages, const char *texturePattern, float aspect)
{
	/* Load Shader */
	if (!LoadShaders())
		return false;

	fAspectRatio = aspect;
	puiTexture = new GLuint[nPages = pages];

	char filename[200];
	for (unsigned int i = pages; i--; )
	{
		sprintf(filename, texturePattern, i);
		if (!LoadTexture(filename, &puiTexture[i]))
		{
			return false;
		}
	}
	uiLeftCurPage = 0;

	UpdateTextures();


	return leftFront.Init(this, -c_fZOffset) && leftBack.Init(this, c_fZOffset) &&
	       rightFront.Init(this, -c_fZOffset) && rightBack.Init(this, c_fZOffset);
}

bool Book::LoadShaders()
{
	const char *aszAttribs[] = { "inVertex", "inTexCoord" }; 
	for (unsigned int i = Page::NumShaders; i--; )
	{
		Page::Shader &program = asProgram[i];

		// Load shader with specified defines
		if (!LoadShader("data/PageFolding.vsh", "data/PageFolding.fsh",
			aszAttribs, 2, c_aszEffectDefines[i], c_aiEffectDefinesNum[i],
			program.vs, program.fs, program.id))
		{
			return false;
		}
		glUniform1i(glGetUniformLocation(program.id, "sTexture"), 0);

		// Store uniform locations for later usage
		program.RadiusLoc = glGetUniformLocation(program.id, "Radius");
		program.DirectionLoc = glGetUniformLocation(program.id, "Direction");
		program.TangentLoc = glGetUniformLocation(program.id, "Tangent");
		program.PointLoc = glGetUniformLocation(program.id, "Point");
		program.AspectRatioLoc = glGetUniformLocation(program.id, "AspectRatio");
		program.ZOffsetLoc = glGetUniformLocation(program.id, "ZOffset");
		program.FlipTextureLoc = glGetUniformLocation(program.id, "FlipTexture");
		program.InputScaleLoc = glGetUniformLocation(program.id, "InputScale");
		program.InputTranslationLoc = glGetUniformLocation(program.id, "InputTranslation");
		program.RotateLoc = glGetUniformLocation(program.id, "Rotate");
	}
	return true;
}

// update texture indices
void Book::UpdateTextures()
{
	leftFront.SetTextures(puiTexture[IMod(uiLeftCurPage, nPages)], 
						puiTexture[IMod(uiLeftCurPage - 1, nPages)]);

	leftBack.SetTextures(puiTexture[IMod(uiLeftCurPage - 2, nPages)], 
						puiTexture[IMod(uiLeftCurPage - 3, nPages)]);


	rightFront.SetTextures(puiTexture[IMod(uiLeftCurPage + 1, nPages)], 
						puiTexture[IMod(uiLeftCurPage + 2, nPages)]);

	rightBack.SetTextures(puiTexture[IMod(uiLeftCurPage + 3, nPages)], 
						puiTexture[IMod(uiLeftCurPage + 4, nPages)]);
}

bool Book::Input(float t, float dt, const TouchState *pTouchState)
{
	bool bLeftRet = leftFront.Input(t, dt, pTouchState);
	bool bRightRet = rightFront.Input(t, dt, pTouchState);

	// Left fold complete
	if (!bLeftRet)
	{
		uiLeftCurPage = IMod(uiLeftCurPage - 2, nPages);
		UpdateTextures();
	}	
	// Right fold complete
	if (!bRightRet)
	{
		uiLeftCurPage = IMod(uiLeftCurPage + 2, nPages);
		UpdateTextures();
	}

	/* */
	// Determine whether this frame should be rendered
	bRenderThisFrame = leftFront.RenderThisFrame() || rightFront.RenderThisFrame() || !bRenderOnDemandInit || !bRenderOnDemand;
	// First frame (or toggle between wireframe and fill): must render
	if (!bRenderOnDemandInit)
		bRenderOnDemandInit = true;
	// If render on demand is not enabled, cond will always evaluate to true, which is fine
	if (bRenderThisFrame)
		uiFilledBuffers = 1;
	// Since we don't have control over glSwapBuffers, we need to fill all the buffers before
	// stopping drawing
	if (!bRenderThisFrame && uiFilledBuffers < uiNumSwapBuffers)
	{
		uiFilledBuffers++;
		bRenderThisFrame = true;
	}
	/* */
	return true;
}


void Book::Render() const
{
	if (!RenderThisFrame())
		return;

	// Clears the color and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// If a page is unfolded, the one behind is not visible and does
	// not need to be rendered.
	if (rightFront.GetFoldValue() == 0.0f)
	{
		rightFront.Render(Page::PageInside);
	}
	// If a page is folded, render the one underneath first.
	else
	{
		rightBack.Render(Page::PageInside);
		rightFront.Render(Page::PageInside);
	}
	// Same as above for left pages
	if (leftFront.GetFoldValue() == 0.0f)
	{
		leftFront.Render(Page::PageInside);
	}
	else
	{
		leftBack.Render(Page::PageInside);
		leftFront.Render(Page::PageInside);
	}

	// If necessary (page is being folded), render the external border
	glDisable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	if (leftFront.GetFoldValue() != 0.0f)
	{
		leftFront.RenderSide(Page::PageOutside);
	}
	if (rightFront.GetFoldValue() != 0.0f)
	{
		rightFront.RenderSide(Page::PageOutside);
	}
	glDisable(GL_BLEND);
}

void Book::RenderDebug() const
{
	leftFront.RenderDebug();
	rightFront.RenderDebug();
}
