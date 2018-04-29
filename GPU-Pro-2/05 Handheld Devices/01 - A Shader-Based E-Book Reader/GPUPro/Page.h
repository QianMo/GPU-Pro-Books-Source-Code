/*!****************************************************************************
 @File          Page.h

 @Title         EBook Demo

 @Copyright     Copyright 2010 by Imagination Technologies Limited.

 @Platform      Independent

 @Description   

******************************************************************************/

#ifndef _PAGE_H_
#define _PAGE_H_

#include "OGLES2Tools.h"
#include "MDKInput.h"
#include "MDKTouch.h"

using namespace MDK::Input;

// Forward declarations
class Book;
class PVRShell;

//#define DEBUG_MODE

//!  Class for rendering a single page
/*!
  Page provides the functionality for rendering a single page based on the user
  input. Page can be used directly to render a fullscreen page.
*/
class Page
{
public:
	//! List of shaders
	enum PageShaders
	{
		ShaderFolding,
		ShaderFoldingAA,
		ShaderFoldingIntBorder,
		ShaderFoldingIntBorderAA,
		ShaderFoldingExtBorder,
		ShaderFoldingWireframe,
		ShaderNoFolding,
		ShaderNoFoldingIntBorder,
		//ShaderNoFoldingIntBorderAA,
		ShaderNoFoldingWireframe,
		NumShaders,
	};

	//! Enumeration to tell which parts of the page need to be rendered
	enum PageMode {
		PageInside = 0x01,
		PageOutside = 0x02,
		PageInsideOutside = (PageInside | PageOutside)
	};

	//! State can be idle or unfolding (automatic without user interaction)
	enum PageState {
		StateIdle,
		StateUnfolding
	};

	//! Struct representing a shader program and related uniforms
	struct Shader
	{
		GLuint id;
		GLuint vs;
		GLuint fs;

		GLint RadiusLoc;
		GLint DirectionLoc;
		GLint TangentLoc;
		GLint PointLoc;
		GLint AspectRatioLoc;
		GLint ZOffsetLoc;
		GLint FlipTextureLoc;
		GLint InputScaleLoc;
		GLint InputTranslationLoc;
		GLint RotateLoc;
	};
private:
	
	//! Front texture
	GLuint uiFrontTex;
	//! Back texture
	GLuint uiBackTex;
	//! Current texture. Used to set "FlipTexture" uniform.
	// See Render(PageMode pageMode) for why this is mutable
	mutable GLint uiCurrTex;

	//! Index to POD file used by this page
	int iPOD;

	//! Indices to meshes within POD file to be used by this page
	CPVRTModelPOD scene;
	int piMeshIndex[4];
	GLuint m_puiVbo[4];
	GLuint m_puiIndexVbo[4];

	//! Scale value for this page. This is set in Init() via a call to the virtual ScaleVector()
	PVRTVec2 vScale;
	//! Translation value for this page. This is set in Init() via a call to the virtual TranslationVector()
	PVRTVec2 vTranslation;

	//! Origin of user interaction. Corresponds to Press event
	PVRTVec2 vOrigin;
	//! End of user interaction. Corresponds to Release event
	PVRTVec2 vEnd;
	//! Whether input gesture was valid or not for a page fold
	bool bValidInput;

	//! Folding direction
	PVRTVec2 vDirection;
	//! Folding tangent (precalculated from direction)
	PVRTVec2 vTangent;
	//! Point on bending axis (calculated in Input())
	PVRTVec2 vPoint;

	//! boolean flag, true if current frame needs to be rendered (set by Input())
	bool bRender;

	//! Current page state
	PageState eState;

protected:
	//! Pointer to parent class
	Book *pParent;

	//! Actual direction passed to the shader (this is derived from vDirection)
	PVRTVec2 vDir2;
	//! depth value offset used to prevent z-fighting between front and back pages
	float fZOffset;

	// State
	float fAngle, fCorrAngle;
	float fFold, fCorrFold;
	float fRadius, fCorrRadius;

	//! Updates vDir2 and fAngle
	virtual void UpdateState(const PVRTVec2 &dir, const float angle);
	// Directly calculates direction from angle
	void UpdateDirection(float angle);

	//! Determines the current corner depending on direction
	void UpdateCorner(PVRTVec2 &corner);

	// Virtuals: these are overridden by HalfPage LeftPage, RightPage
	virtual const PVRTVec2 ScaleVector() const { return PVRTVec2(1.0f, 1.0f); }
	virtual const PVRTVec2 TranslationVector() const { return PVRTVec2(0.0f, 0.0f); }

	virtual bool CanFold(float xDir) const { return true; }
	virtual bool UpdateAngleAndRadius() { return false; }
	virtual bool PageTurn() { return false; }
	virtual bool PageTurnComplete() { return false; }
	//! Tells whether the user is dragging from a valid area of the page
	virtual bool InputCondition(const TouchState *pTouchState) { return true; }

	virtual bool AngleSnap(float angle, float corrAngle) const { return false; }
	float ClampRadius();

	virtual GLenum BackFace() const { return GL_BACK; }
	virtual GLenum FrontFace() const { return GL_FRONT; }

	void SetupShader(const Shader *shader) const;

	// String pointing to resource pod file to be used (derived classes can
	// use different POD files due to different tessellation)
	virtual const char *GetPODName() const { return "data/grid.POD"; };

	//! Get desired shader struct 
	const Shader *GetShader(PageShaders shader) const;

	bool LoadVbos(CPVRTModelPOD &scene);
	void DrawMesh(int iVBO, bool wireframe) const;
public:
	Page();
	virtual ~Page() { }

	//! Initializes page by setting values from virtuals and loading geomerty from
	// appropriate POD file
	virtual bool Init(Book *parent, float zOffset);
	
	//! Processes the state update.
	virtual bool Input(float t, float dt, const TouchState *pTouchState);
	//! Render one side of the page. This is public to allow manual rendering from outside this class
	virtual void RenderSide(PageMode pageMode) const;
	//! Main rendering function
	virtual void Render(PageMode pageMode) const;

	//! Render origin->end line
	virtual void RenderDebug() const;

	// Sets the new texture indices for front and back
	void SetTextures(GLuint front, GLuint back);

	const float GetFoldValue() const { return fFold; }
	const float GetAngle() const { return fAngle; }
	const float GetRadius() const { return fRadius; }

	//! See Input() for bRender update policy
	const bool RenderThisFrame() const { return bRender; }
};

//!  Class for rendering a single page of width half of the screen
/*!
  HalfPage adds the required functionality to Page in order to render a page
  half the width of the original one.
*/

class HalfPage : public Page
{
protected:
	virtual void UpdateState(const PVRTVec2 &dir, const float angle);
	virtual const char *GetPODName() const = 0;

	virtual const PVRTVec2 ScaleVector() const { return PVRTVec2(0.5f, 1.0f); }

	virtual bool PageTurn();
public:
	virtual bool PageTurnComplete();
};

class LeftPage : public HalfPage
{
protected:

	virtual const char *GetPODName() const { return "data/grid-left.POD"; }

	virtual bool InputCondition(const TouchState *pTouchState);
	virtual const PVRTVec2 TranslationVector() const { return PVRTVec2(-0.5f, 0.0f); }

	virtual bool CanFold(float xDir) const;
	virtual bool UpdateAngleAndRadius();
	virtual bool AngleSnap(float angle, float corrAngle) const;

	virtual GLenum BackFace() const { return GL_BACK; }
	virtual GLenum FrontFace() const { return GL_FRONT; }

public:
	LeftPage();

};

class RightPage : public HalfPage
{
protected:

	virtual const char *GetPODName() const { return "data/grid-right.POD"; }

	virtual bool InputCondition(const TouchState *pTouchState);
	virtual const PVRTVec2 TranslationVector() const { return PVRTVec2(0.5f, 0.0f); }

	virtual bool CanFold(float xDir) const;
	virtual bool UpdateAngleAndRadius();
	virtual bool AngleSnap(float angle, float corrAngle) const;

	virtual GLenum BackFace() const { return GL_FRONT; }
	virtual GLenum FrontFace() const { return GL_BACK; }

public:
	RightPage();
};

class Book
{

protected:
	//! Pointer to parent
	PVRShell *pShell;

	//! Array of shader programs
	Page::Shader asProgram[Page::NumShaders];

	//! Number of textures/pages
	unsigned int nPages;
	//! Dynamically allocated array of texture IDs
	GLuint *puiTexture;
	//! Index within array to current left page. This is updated each time
	// a page is folded
	unsigned int uiLeftCurPage;

	// Variables accessed by Page istances
	//! Whether to render in wireframe mode
	bool bWireframe;
	//! Aspect ratio
	float fAspectRatio;
	//! Whether to use procedural antialiasing
	bool bUseAA;
	//! Whether to use render on demand
	bool bRenderOnDemand;
	//! On the first frame the rendering is always performed, then
	// only on demand. This is a flag to tell whether we are on the
	// first frame.
	bool bRenderOnDemandInit;

	//! Number of swap buffers
	unsigned int uiNumSwapBuffers;
	//! Numbef of filled buffers
	unsigned int uiFilledBuffers;

	//! This is set inside Input() to tell whether the frame should be rendered
	bool bRenderThisFrame;

	// only four pages can be visible at a time
	LeftPage leftFront, leftBack;
	RightPage rightFront, rightBack;

	//! If any of the pages completes the fold, all texture IDs need to be updated
	// in the other pages
	void UpdateTextures();
	//! Called by Init() to load shaders. Pages can get access to a shader via GetShader()
	bool LoadShaders();
public:
	Book(PVRShell *shell);
	~Book();

	//! Loads shaders and textures
	bool Init(unsigned int pages, const char *texturePattern, float aspect);

	//! Changes Render on demand state
	//bool SetRenderOnDemand(const bool value) { return (bRenderOnDemand = value); }

	//! Updates pages state
	bool Input(float t, float dt, const TouchState *pTouchState);

	//! Rendering function that draws all pages
	void Render() const;

	//! Renders directions on pages
	void RenderDebug() const;


	// Some getters
	//! Gets required shader (can be called by child pages)
	const Page::Shader *GetShader(Page::PageShaders shader) const { return &asProgram[shader]; }
	const bool Wireframe() const { return bWireframe; }
	const float AspectRatio() const { return Rotated() ? 1.0f / fAspectRatio : fAspectRatio; }
	const int Rotated() const { return pShell->PVRShellGet(prefIsRotated) ? 1 : 0; }
	const float AAEnabled() const { return bUseAA; }
	const Page *GetLeftPage() const { return &leftFront; }
	const Page *GetRightPage() const { return &rightFront; }
	const bool IsRenderOnDemand() const { return bRenderOnDemand; }

	// If render on demand is enabled, this returns true only if the state
	// of any of the pages changed (i.e. needs redraw)
	inline bool RenderThisFrame() const { return bRenderThisFrame; }

	//! Toggle Render on demand option
	bool ToggleRenderOnDemand() { return (bRenderOnDemand = !bRenderOnDemand); }

	//! Force to render this frame
	void ForceRender() { bRenderOnDemandInit = false; }

	//! Toggle wireframe
	inline bool ToggleWireframe()
	{
		ForceRender();
		return (bWireframe = !bWireframe);
	}
		
};

#endif
