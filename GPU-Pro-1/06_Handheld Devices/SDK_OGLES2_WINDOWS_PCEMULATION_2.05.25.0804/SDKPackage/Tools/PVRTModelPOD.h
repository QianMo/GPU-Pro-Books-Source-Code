/******************************************************************************

 @File         PVRTModelPOD.h

 @Title        PVRTModelPOD

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Code to load POD files - models exported from MAX.

******************************************************************************/
#ifndef _PVRTMODELPOD_H_
#define _PVRTMODELPOD_H_

#include "PVRTVector.h"
#include "PVRTError.h"
#include "PVRTVertex.h"
#include "PVRTBoneBatch.h"

/****************************************************************************
** Defines
****************************************************************************/
#define PVRTMODELPOD_VERSION	("AB.POD.2.0") /*!< POD file version string */

// PVRTMODELPOD Scene Flags
#define PVRTMODELPODSF_FIXED	(0x00000001)   /*!< PVRTMODELPOD Fixed-point 16.16 data (otherwise float) flag */

/****************************************************************************
** Enumerations
****************************************************************************/
/*!****************************************************************************
 @Struct      EPODLight
 @Brief       Enum for the POD format light types
******************************************************************************/
enum EPODLight
{
	ePODPoint=0,	 /*!< Point light */
	ePODDirectional, /*!< Directional light */
	ePODSpot		 /*!< Spot light */
};

/*!****************************************************************************
 @Struct      EPODPrimitiveType
 @Brief       Enum for the POD format primitive types
******************************************************************************/
enum EPODPrimitiveType
{
	ePODTriangles=0, /*!< Triangles */
	ePODLines		 /*!< Lines*/
};

/*!****************************************************************************
 @Struct      EPODAnimationData
 @Brief       Enum for the POD format animation types
******************************************************************************/
enum EPODAnimationData
{
	ePODHasPositionAni	= 0x01,	/*!< Position animation */
	ePODHasRotationAni	= 0x02, /*!< Rotation animation */
	ePODHasScaleAni		= 0x04, /*!< Scale animation */
	ePODHasMatrixAni	= 0x08  /*!< Matrix animation */
};

/*!****************************************************************************
 @Struct      EPODMaterialFlags
 @Brief       Enum for the material flag options
******************************************************************************/
enum EPODMaterialFlags
{
	ePODEnableBlending	= 0x01,	/*!< Enable blending for this material */
};

/*!****************************************************************************
 @Struct      EPODBlendFunc
 @Brief       Enum for the POD format blend functions
******************************************************************************/
enum EPODBlendFunc
{
	ePODBlendFunc_ZERO,
	ePODBlendFunc_ONE,
	ePODBlendFunc_BLEND_FACTOR,
	ePODBlendFunc_ONE_MINUS_BLEND_FACTOR,

	ePODBlendFunc_SRC_COLOR = 0x0300,
	ePODBlendFunc_ONE_MINUS_SRC_COLOR,
	ePODBlendFunc_SRC_ALPHA,
	ePODBlendFunc_ONE_MINUS_SRC_ALPHA,
	ePODBlendFunc_DST_ALPHA,
	ePODBlendFunc_ONE_MINUS_DST_ALPHA,
	ePODBlendFunc_DST_COLOR,
	ePODBlendFunc_ONE_MINUS_DST_COLOR,
	ePODBlendFunc_SRC_ALPHA_SATURATE,

	ePODBlendFunc_CONSTANT_COLOR = 0x8001,
	ePODBlendFunc_ONE_MINUS_CONSTANT_COLOR,
	ePODBlendFunc_CONSTANT_ALPHA,
	ePODBlendFunc_ONE_MINUS_CONSTANT_ALPHA
};

/*!****************************************************************************
 @Struct      EPODBlendOp
 @Brief       Enum for the POD format blend operation
******************************************************************************/
enum EPODBlendOp
{
	ePODBlendOp_ADD = 0x8006,
	ePODBlendOp_MIN,
	ePODBlendOp_MAX,
	ePODBlendOp_SUBTRACT = 0x800A,
	ePODBlendOp_REVERSE_SUBTRACT,
};

/****************************************************************************
** Structures
****************************************************************************/
/*!****************************************************************************
 @Class      CPODData
 @Brief      A class for representing POD data
******************************************************************************/
class CPODData {
public:
	/*!***************************************************************************
	@Function			Reset
	@Description		Resets the POD Data to NULL
	*****************************************************************************/
	void Reset();

public:
	EPVRTDataType	eType;		/*!< Type of data stored */
	unsigned int	n;			/*!< Number of values per vertex */
	unsigned int	nStride;	/*!< Distance in bytes from one array entry to the next */
	unsigned char	*pData;		/*!< Actual data (array of values); if mesh is interleaved, this is an OFFSET from pInterleaved */
};

/*!****************************************************************************
 @Struct      SPODCamera
 @Brief       Struct for storing POD camera data
******************************************************************************/
struct SPODCamera {
	int			nIdxTarget;			/*!< Index of the target object */
	VERTTYPE	fFOV;				/*!< Field of view */
	VERTTYPE	fFar;				/*!< Far clip plane */
	VERTTYPE	fNear;				/*!< Near clip plane */
	VERTTYPE	*pfAnimFOV;			/*!< 1 VERTTYPE per frame of animation. */
};

/*!****************************************************************************
 @Struct      SPODLight
 @Brief       Struct for storing POD light data
******************************************************************************/
struct SPODLight {
	int			nIdxTarget;		/*!< Index of the target object */
	VERTTYPE	pfColour[3];	/*!< Light colour (0.0f -> 1.0f for each channel) */
	EPODLight	eType;			/*!< Light type (point, directional, spot etc.) */
	float		fConstantAttenuation;	/*!< Constant attenuation */
	float		fLinearAttenuation;		/*!< Linear atternuation */
	float		fQuadraticAttenuation;	/*!< Quadratic attenuation */
	float		fFalloffAngle;			/*!< Falloff angle (in radians) */
	float		fFalloffExponent;		/*!< Falloff exponent */
};

/*!****************************************************************************
 @Struct      SPODMesh
 @Brief       Struct for storing POD mesh data
******************************************************************************/
struct SPODMesh {
	unsigned int		nNumVertex;		/*!< Number of vertices in the mesh */
	unsigned int		nNumFaces;		/*!< Number of triangles in the mesh */
	unsigned int		nNumUVW;		/*!< Number of texture coordinate channels per vertex */
	CPODData			sFaces;			/*!< List of triangle indices */
	unsigned int		*pnStripLength;	/*!< If mesh is stripped: number of tris per strip. */
	unsigned int		nNumStrips;		/*!< If mesh is stripped: number of strips, length of pnStripLength array. */
	CPODData			sVertex;		/*!< List of vertices (x0, y0, z0, x1, y1, z1, x2, etc...) */
	CPODData			sNormals;		/*!< List of vertex normals (Nx0, Ny0, Nz0, Nx1, Ny1, Nz1, Nx2, etc...) */
	CPODData			sTangents;		/*!< List of vertex tangents (Tx0, Ty0, Tz0, Tx1, Ty1, Tz1, Tx2, etc...) */
	CPODData			sBinormals;		/*!< List of vertex binormals (Bx0, By0, Bz0, Bx1, By1, Bz1, Bx2, etc...) */
	CPODData			*psUVW;			/*!< List of UVW coordinate sets; size of array given by 'nNumUVW' */
	CPODData			sVtxColours;	/*!< A colour per vertex */
	CPODData			sBoneIdx;		/*!< nNumBones*nNumVertex ints (Vtx0Idx0, Vtx0Idx1, ... Vtx1Idx0, Vtx1Idx1, ...) */
	CPODData			sBoneWeight;	/*!< nNumBones*nNumVertex floats (Vtx0Wt0, Vtx0Wt1, ... Vtx1Wt0, Vtx1Wt1, ...) */

	unsigned char		*pInterleaved;	/*!< Interleaved vertex data */

	CPVRTBoneBatches	sBoneBatches;	/*!< Bone tables */

	EPODPrimitiveType	ePrimitiveType;	/*!< Primitive type used by this mesh */
};

/*!****************************************************************************
 @Struct      SPODNode
 @Brief       Struct for storing POD node data
******************************************************************************/
struct SPODNode {
	int			 nIdx;				/*!< Index into mesh, light or camera array, depending on which object list contains this Node */
	char		 *pszName;			/*!< Name of object */
	int			 nIdxMaterial;		/*!< Index of material used on this mesh */

	int			 nIdxParent;		/*!< Index into MeshInstance array; recursively apply ancestor's transforms after this instance's. */

	unsigned int nAnimFlags;		/*!< Stores which animation arrays the POD Node contains */
	VERTTYPE	*pfAnimPosition;	/*!< 3 floats per frame of animation. */
	VERTTYPE	*pfAnimRotation;	/*!< 4 floats per frame of animation. */
	VERTTYPE	*pfAnimScale;		/*!< 7 floats per frame of animation. */
	VERTTYPE	*pfAnimMatrix;		/*!< 16 floats per frame of animation. */
};

/*!****************************************************************************
 @Struct      SPODTexture
 @Brief       Struct for storing POD texture data
******************************************************************************/
struct SPODTexture {
	char	*pszName;			/*!< File-name of texture */
};

/*!****************************************************************************
 @Struct      SPODMaterial
 @Brief       Struct for storing POD material data
******************************************************************************/
struct SPODMaterial {
	char		*pszName;				/*!< Name of material */
	int			nIdxTexDiffuse;			/*!< Idx into pTexture for the diffuse texture */
	int			nIdxTexAmbient;			/*!< Idx into pTexture for the ambient texture */
	int			nIdxTexSpecularColour;	/*!< Idx into pTexture for the specular colour texture */
	int			nIdxTexSpecularLevel;	/*!< Idx into pTexture for the specular level texture */
	int			nIdxTexBump;			/*!< Idx into pTexture for the bump map */
	int			nIdxTexEmissive;		/*!< Idx into pTexture for the emissive texture */
	int			nIdxTexGlossiness;		/*!< Idx into pTexture for the glossiness texture */
	int			nIdxTexOpacity;			/*!< Idx into pTexture for the opacity texture */
	int			nIdxTexReflection;		/*!< Idx into pTexture for the reflection texture */
	int			nIdxTexRefraction;		/*!< Idx into pTexture for the refraction texture */
	VERTTYPE	fMatOpacity;			/*!< Material opacity (used with vertex alpha ?) */
	VERTTYPE	pfMatAmbient[3];		/*!< Ambient RGB value */
	VERTTYPE	pfMatDiffuse[3];		/*!< Diffuse RGB value */
	VERTTYPE	pfMatSpecular[3];		/*!< Specular RGB value */
	VERTTYPE	fMatShininess;			/*!< Material shininess */
	char		*pszEffectFile;			/*!< Name of effect file */
	char		*pszEffectName;			/*!< Name of effect in the effect file */

	EPODBlendFunc	eBlendSrcRGB;		/*!< Blending RGB source value */
	EPODBlendFunc	eBlendSrcA;			/*!< Blending alpha source value */
	EPODBlendFunc	eBlendDstRGB;		/*!< Blending RGB destination value */
	EPODBlendFunc	eBlendDstA;			/*!< Blending alpha destination value */
	EPODBlendOp		eBlendOpRGB;		/*!< Blending RGB operation */
	EPODBlendOp		eBlendOpA;			/*!< Blending alpha operation */
	VERTTYPE		pfBlendColour[4];	/*!< A RGBA colour to be used in blending */
	VERTTYPE		pfBlendFactor[4];	/*!< An array of blend factors, one for each RGBA component */

	unsigned int	nFlags;				/*!< Stores information about the material e.g. Enable blending */
};

/*!****************************************************************************
 @Struct      SPODScene
 @Brief       Struct for storing POD scene data
******************************************************************************/
struct SPODScene {
	VERTTYPE	pfColourBackground[3];		/*!< Background colour */
	VERTTYPE	pfColourAmbient[3];			/*!< Ambient colour */

	unsigned int	nNumCamera;				/*!< The length of the array pCamera */
	SPODCamera		*pCamera;				/*!< Camera nodes array */

	unsigned int	nNumLight;				/*!< The length of the array pLight */
	SPODLight		*pLight;				/*!< Light nodes array */

	unsigned int	nNumMesh;				/*!< The length of the array pMesh */
	SPODMesh		*pMesh;					/*!< Mesh array. Meshes may be instanced several times in a scene; i.e. multiple Nodes may reference any given mesh. */

	unsigned int	nNumNode;		/*!< Number of items in the array pNode */
	unsigned int	nNumMeshNode;	/*!< Number of items in the array pNode which are objects */
	SPODNode		*pNode;			/*!< Node array. Sorted as such: objects, lights, cameras, Everything Else (bones, helpers etc) */

	unsigned int	nNumTexture;	/*!< Number of textures in the array pTexture */
	SPODTexture		*pTexture;		/*!< Texture array */

	unsigned int	nNumMaterial;	/*!< Number of materials in the array pMaterial */
	SPODMaterial	*pMaterial;		/*!< Material array */

	unsigned int	nNumFrame;		/*!< Number of frames of animation */
	unsigned int	nFlags;			/*!< PVRTMODELPODSF_* bit-flags */
};

struct SPVRTPODImpl;	// Internal implementation data

/*!***************************************************************************
@Class CPVRTModelPOD
@Brief A class for loading and storing data from POD files/headers
*****************************************************************************/
class CPVRTModelPOD : public SPODScene{
public:
	/*!***************************************************************************
	 @Function		Constructor
	 @Description	Constructor for CPVRTModelPOD class
	*****************************************************************************/
	CPVRTModelPOD();

	/*!***************************************************************************
	 @Function		Destructor
	 @Description	Destructor for CPVRTModelPOD class
	*****************************************************************************/
	~CPVRTModelPOD();

	/*!***************************************************************************
	@Function			ReadFromFile
	@Input				pszFileName		Filename to load
	@Output			pszExpOpt		String in which to place exporter options
	@Input				count			Maximum number of characters to store.
	@Output			pszHistory		String in which to place the pod file history
	@Input				historyCount	Maximum number of characters to store.
	@Return			PVR_SUCCESS if successful, PVR_FAIL if not
	@Description		Loads the specified ".POD" file; returns the scene in
						pScene. This structure must later be destroyed with
						PVRTModelPODDestroy() to prevent memory leaks.
						".POD" files are exported using the PVRGeoPOD exporters.
						If pszExpOpt is NULL, the scene is loaded; otherwise the
						scene is not loaded and pszExpOpt is filled in. The same
						is true for pszHistory.
	*****************************************************************************/
	EPVRTError ReadFromFile(
		const char		* const pszFileName,
		char			* const pszExpOpt = NULL,
		const size_t	count = 0,
		char			* const pszHistory = NULL,
		const size_t	historyCount = 0);

	/*!***************************************************************************
	@Function			ReadFromMemory
	@Input				pData			Data to load
	@Input				i32Size			Size of data
	@Output			pszExpOpt		String in which to place exporter options
	@Input				count			Maximum number of characters to store.
	@Output			pszHistory		String in which to place the pod file history
	@Input				historyCount	Maximum number of characters to store.
	@Return			PVR_SUCCESS if successful, PVR_FAIL if not
	@Description		Loads the supplied pod data. This data can be exported
						directly to a header using one of the pod exporters.
						If pszExpOpt is NULL, the scene is loaded; otherwise the
						scene is not loaded and pszExpOpt is filled in. The same
						is true for pszHistory.
	*****************************************************************************/
	EPVRTError ReadFromMemory(
		const char		* pData,
		const size_t	i32Size,
		char			* const pszExpOpt = NULL,
		const size_t	count = NULL,
		char			* const pszHistory = NULL,
		const size_t	historyCount = NULL);

	/*!***************************************************************************
	 @Function		ReadFromMemory
	 @Input			scene			Scene data from the header file
	 @Return		PVR_SUCCESS if successful, PVR_FAIL if not
	 @Description	Sets the scene data from the supplied data structure. Use
					when loading from .H files.
	*****************************************************************************/
	EPVRTError ReadFromMemory(
		const SPODScene &scene);

	/*!***************************************************************************
	 @Function		CopyFromMemory
	 @Input			scene			Scene data from the header file
	 @Return		PVR_SUCCESS if successful, PVR_FAIL if not
	 @Description	Copies the scene data from the supplied data structure. Use
					when loading from .H files where you want to modify the data.
	*****************************************************************************/
	EPVRTError CopyFromMemory(
		const SPODScene &scene);

#ifdef WIN32
	/*!***************************************************************************
	 @Function		ReadFromResource
	 @Input			pszName			Name of the resource to load from
	 @Return		PVR_SUCCESS if successful, PVR_FAIL if not
	 @Description	Loads the specified ".POD" file; returns the scene in
					pScene. This structure must later be destroyed with
					PVRTModelPODDestroy() to prevent memory leaks.
					".POD" files are exported from 3D Studio MAX using a
					PowerVR plugin.
	*****************************************************************************/
	EPVRTError ReadFromResource(
		const TCHAR * const pszName);
#endif

	/*!***********************************************************************
	 @Function		InitImpl
	 @Description	Used by the Read*() fns to initialise implementation
					details. Should also be called by applications which
					manually build data in the POD structures for rendering;
					in this case call it after the data has been created.
					Otherwise, do not call this function.
	*************************************************************************/
	EPVRTError InitImpl();

	/*!***********************************************************************
	 @Function		DestroyImpl
	 @Description	Used to free memory allocated by the implementation.
	*************************************************************************/
	void DestroyImpl();

	/*!***********************************************************************
	 @Function		FlushCache
	 @Description	Clears the matrix cache; use this if necessary when you
					edit the position or animation of a node.
	*************************************************************************/
	void FlushCache();
	/*!***************************************************************************
	 @Function		Destroy
	 @Description	Frees the memory allocated to store the scene in pScene.
	*****************************************************************************/
	void Destroy();

	/*!***************************************************************************
	 @Function		SetFrame
	 @Input			fFrame			Frame number
	 @Description	Set the animation frame for which subsequent Get*() calls
					should return data.
	*****************************************************************************/
	void SetFrame(
		const VERTTYPE fFrame);

	/*!***************************************************************************
	 @Function		GetRotationMatrix
	 @Output		mOut			Rotation matrix
	 @Input			node			Node to get the rotation matrix from
	 @Description	Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
	*****************************************************************************/
	void GetRotationMatrix(
		PVRTMATRIX		&mOut,
		const SPODNode	&node) const;

	/*!***************************************************************************
	 @Function		GetRotationMatrix
	 @Input			node			Node to get the rotation matrix from
	 @Returns		Rotation matrix
	 @Description	Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
	*****************************************************************************/
	PVRTMat4 GetRotationMatrix(
		const SPODNode	&node) const;

	/*!***************************************************************************
	 @Function		GetScalingMatrix
	 @Output		mOut			Scaling matrix
	 @Input			node			Node to get the rotation matrix from
	 @Description	Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
	*****************************************************************************/
	void GetScalingMatrix(
		PVRTMATRIX		&mOut,
		const SPODNode	&node) const;

	/*!***************************************************************************
	 @Function		GetScalingMatrix
	 @Input			node			Node to get the rotation matrix from
	 @Returns		Scaling matrix
	 @Description	Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
	*****************************************************************************/
	PVRTMat4 GetScalingMatrix(
		const SPODNode	&node) const;

	/*!***************************************************************************
	 @Function		GetTranslation
	 @Output		V				Translation vector
	 @Input			node			Node to get the translation vector from
	 @Description	Generates the translation vector for the given Mesh
					Instance. Uses animation data.
	*****************************************************************************/
	void GetTranslation(
		PVRTVECTOR3		&V,
		const SPODNode	&node) const;

	/*!***************************************************************************
	 @Function		GetTranslation
	 @Input			node			Node to get the translation vector from
	  @Returns		Translation vector
	 @Description	Generates the translation vector for the given Mesh
					Instance. Uses animation data.
	*****************************************************************************/
	PVRTVec3 GetTranslation(
		const SPODNode	&node) const;

	/*!***************************************************************************
	 @Function		GetTranslationMatrix
	 @Output		mOut			Translation matrix
	 @Input			node			Node to get the translation matrix from
	 @Description	Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
	*****************************************************************************/
	void GetTranslationMatrix(
		PVRTMATRIX		&mOut,
		const SPODNode	&node) const;

	/*!***************************************************************************
	 @Function		GetTranslationMatrix
	 @Input			node			Node to get the translation matrix from
	 @Returns		Translation matrix
	 @Description	Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
	*****************************************************************************/
	PVRTMat4 GetTranslationMatrix(
		const SPODNode	&node) const;

    /*!***************************************************************************
	 @Function		GetTransformationMatrix
	 @Output		mOut			Transformation matrix
	 @Input			node			Node to get the transformation matrix from
	 @Description	Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
	*****************************************************************************/
	void GetTransformationMatrix(PVRTMATRIX &mOut, const SPODNode &node) const;

	/*!***************************************************************************
	 @Function		GetWorldMatrixNoCache
	 @Output		mOut			World matrix
	 @Input			node			Node to get the world matrix from
	 @Description	Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
	*****************************************************************************/
	void GetWorldMatrixNoCache(
		PVRTMATRIX		&mOut,
		const SPODNode	&node) const;

	/*!***************************************************************************
	@Function		GetWorldMatrixNoCache
	@Input			node			Node to get the world matrix from
	@Returns		World matrix
	@Description	Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
	*****************************************************************************/
	PVRTMat4 GetWorldMatrixNoCache(
		const SPODNode	&node) const;

	/*!***************************************************************************
	 @Function		GetWorldMatrix
	 @Output		mOut			World matrix
	 @Input			node			Node to get the world matrix from
	 @Description	Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
	*****************************************************************************/
	void GetWorldMatrix(
		PVRTMATRIX		&mOut,
		const SPODNode	&node) const;

	/*!***************************************************************************
	@Function		GetWorldMatrix
	@Input			node			Node to get the world matrix from
	@Returns		World matrix
	@Description	Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
	*****************************************************************************/
	PVRTMat4 GetWorldMatrix(const SPODNode& node) const;

	/*!***************************************************************************
	 @Function		GetBoneWorldMatrix
	 @Output		mOut			Bone world matrix
	 @Input			NodeMesh		Mesh to take the bone matrix from
	 @Input			NodeBone		Bone to take the matrix from
	 @Description	Generates the world matrix for the given bone.
	*****************************************************************************/
	void GetBoneWorldMatrix(
		PVRTMATRIX		&mOut,
		const SPODNode	&NodeMesh,
		const SPODNode	&NodeBone);

	/*!***************************************************************************
	@Function		GetBoneWorldMatrix
	@Input			NodeMesh		Mesh to take the bone matrix from
	@Input			NodeBone		Bone to take the matrix from
	@Returns		Bone world matrix
	@Description	Generates the world matrix for the given bone.
	*****************************************************************************/
	PVRTMat4 GetBoneWorldMatrix(
		const SPODNode	&NodeMesh,
		const SPODNode	&NodeBone);

	/*!***************************************************************************
	 @Function		GetCamera
	 @Output		vFrom			Position of the camera
	 @Output		vTo				Target of the camera
	 @Output		vUp				Up direction of the camera
	 @Input			nIdx			Camera number
	 @Return		Camera horizontal FOV
	 @Description	Calculate the From, To and Up vectors for the given
					camera. Uses animation data.
					Note that even if the camera has a target, *pvTo is not
					the position of that target. *pvTo is a position in the
					correct direction of the target, one unit away from the
					camera.
	*****************************************************************************/
	VERTTYPE GetCamera(
		PVRTVECTOR3			&vFrom,
		PVRTVECTOR3			&vTo,
		PVRTVECTOR3			&vUp,
		const unsigned int	nIdx) const;

	/*!***************************************************************************
	 @Function		GetCameraPos
	 @Output		vFrom			Position of the camera
	 @Output		vTo				Target of the camera
	 @Input			nIdx			Camera number
	 @Return		Camera horizontal FOV
	 @Description	Calculate the position of the camera and its target. Uses
					animation data.
					If the queried camera does not have a target, *pvTo is
					not changed.
	*****************************************************************************/
	VERTTYPE GetCameraPos(
		PVRTVECTOR3			&vFrom,
		PVRTVECTOR3			&vTo,
		const unsigned int	nIdx) const;

	/*!***************************************************************************
	 @Function		GetLight
	 @Output		vPos			Position of the light
	 @Output		vDir			Direction of the light
	 @Input			nIdx			Light number
	 @Description	Calculate the position and direction of the given Light.
					Uses animation data.
	*****************************************************************************/
	void GetLight(
		PVRTVECTOR3			&vPos,
		PVRTVECTOR3			&vDir,
		const unsigned int	nIdx) const;

	/*!***************************************************************************
	 @Function		GetLightPosition
	 @Input			u32Idx			Light number
	 @Return		PVRTVec4 position of light with w set correctly
	 @Description	Calculate the position the given Light. Uses animation data.
	*****************************************************************************/
	PVRTVec4 GetLightPosition(const unsigned int u32Idx) const;

	/*!***************************************************************************
	@Function		GetLightDirection
	@Input			u32Idx			Light number
	@Return			PVRTVec4 direction of light with w set correctly
	@Description	Calculate the direction of the given Light. Uses animation data.
	*****************************************************************************/
	PVRTVec4 GetLightDirection(const unsigned int u32Idx) const;

	/*!***************************************************************************
	 @Function		CreateSkinIdxWeight
	 @Output		pIdx				Four bytes containing matrix indices for vertex (0..255) (D3D: use UBYTE4)
	 @Output		pWeight				Four bytes containing blend weights for vertex (0.0 .. 1.0) (D3D: use D3DCOLOR)
	 @Input			nVertexBones		Number of bones this vertex uses
	 @Input			pnBoneIdx			Pointer to 'nVertexBones' indices
	 @Input			pfBoneWeight		Pointer to 'nVertexBones' blend weights
	 @Description	Creates the matrix indices and blend weights for a boned
					vertex. Call once per vertex of a boned mesh.
	*****************************************************************************/
	EPVRTError CreateSkinIdxWeight(
		char			* const pIdx,
		char			* const pWeight,
		const int		nVertexBones,
		const int		* const pnBoneIdx,
		const VERTTYPE	* const pfBoneWeight);

	/*!***************************************************************************
	 @Function		SavePOD
	 @Input			pszFilename		Filename to save to
	 @Input			pszExpOpt		A string containing the options used by the exporter
	 @Input			pszHistory		A string containing the history of the exported pod file
	 @Description	Save a binary POD file (.POD).
	*****************************************************************************/
	EPVRTError SavePOD(const char * const pszFilename, const char * const pszExpOpt = 0, const char * const pszHistory = 0);

private:
	SPVRTPODImpl	*m_pImpl;	/*!< Internal implementation data */
};

/****************************************************************************
** Declarations
****************************************************************************/

/*!***************************************************************************
 @Function		PVRTModelPODDataTypeSize
 @Input			type		Type to get the size of
 @Return		Size of the data element
 @Description	Returns the size of each data element.
*****************************************************************************/
size_t PVRTModelPODDataTypeSize(const EPVRTDataType type);

/*!***************************************************************************
 @Function		PVRTModelPODDataTypeComponentCount
 @Input			type		Type to get the number of components from
 @Return		number of components in the data element
 @Description	Returns the number of components in a data element.
*****************************************************************************/
size_t PVRTModelPODDataTypeComponentCount(const EPVRTDataType type);

/*!***************************************************************************
 @Function		PVRTModelPODDataStride
 @Input			data		Data elements
 @Return		Size of the vector elements
 @Description	Returns the size of the vector of data elements.
*****************************************************************************/
size_t PVRTModelPODDataStride(const CPODData &data);

/*!***************************************************************************
 @Function		PVRTModelPODDataConvert
 @Modified		data		Data elements to convert
 @Input			eNewType	New type of elements
 @Input			nCnt		Number of elements
 @Description	Convert the format of the array of vectors.
*****************************************************************************/
void PVRTModelPODDataConvert(CPODData &data, const unsigned int nCnt, const EPVRTDataType eNewType);

/*!***************************************************************************
 @Function		PVRTModelPODDataShred
 @Modified		data		Data elements to modify
 @Input			nCnt		Number of elements
 @Input			nMask		Channel masks
 @Description	Reduce the number of dimensions in 'data' using the channel
				masks in 'nMask'.
*****************************************************************************/
void PVRTModelPODDataShred(CPODData &data, const unsigned int nCnt, const unsigned int nMask);

/*!***************************************************************************
 @Function		PVRTModelPODToggleInterleaved
 @Modified		mesh		Mesh to modify
 @Description	Switches the supplied mesh to or from interleaved data format.
*****************************************************************************/
void PVRTModelPODToggleInterleaved(SPODMesh &mesh);

/*!***************************************************************************
 @Function		PVRTModelPODDeIndex
 @Modified		mesh		Mesh to modify
 @Description	De-indexes the supplied mesh. The mesh must be
				Interleaved before calling this function.
*****************************************************************************/
void PVRTModelPODDeIndex(SPODMesh &mesh);

/*!***************************************************************************
 @Function		PVRTModelPODToggleStrips
 @Modified		mesh		Mesh to modify
 @Description	Converts the supplied mesh to or from strips.
*****************************************************************************/
void PVRTModelPODToggleStrips(SPODMesh &mesh);

/*!***************************************************************************
 @Function		PVRTModelPODCountIndices
 @Input			mesh		Mesh
 @Return		Number of indices used by mesh
 @Description	Counts the number of indices of a mesh
*****************************************************************************/
unsigned int PVRTModelPODCountIndices(const SPODMesh &mesh);

/*!***************************************************************************
 @Function		PVRTModelPODToggleFixedPoint
 @Modified		s		Scene to modify
 @Description	Switch all non-vertex data between fixed-point and
				floating-point.
*****************************************************************************/
void PVRTModelPODToggleFixedPoint(SPODScene &s);

#endif /* _PVRTMODELPOD_H_ */

/*****************************************************************************
 End of file (PVRTModelPOD.h)
*****************************************************************************/
