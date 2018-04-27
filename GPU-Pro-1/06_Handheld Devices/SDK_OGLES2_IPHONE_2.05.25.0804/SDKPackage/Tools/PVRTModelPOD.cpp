/******************************************************************************

 @File         PVRTModelPOD.cpp

 @Title        PVRTModelPOD

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Code to load POD files - models exported from MAX.

******************************************************************************/
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "PVRTGlobal.h"
#include "PVRTContext.h"
#include "PVRTFixedPoint.h"
#include "PVRTMatrix.h"
#include "PVRTQuaternion.h"
#include "PVRTVertex.h"
#include "PVRTBoneBatch.h"
#include "PVRTModelPOD.h"
#include "PVRTMisc.h"
#include "PVRTResourceFile.h"

/****************************************************************************
** Defines
****************************************************************************/
#define PVRTMODELPOD_TAG_MASK			(0x80000000)
#define PVRTMODELPOD_TAG_START			(0x00000000)
#define PVRTMODELPOD_TAG_END			(0x80000000)

#define CFAH		(1024)

/****************************************************************************
** Enumerations
****************************************************************************/
/*!****************************************************************************
 @Struct      EPODFileName
 @Brief       Enum for the binary pod blocks
******************************************************************************/
enum EPODFileName
{
	ePODFileVersion				= 1000,
	ePODFileScene,
	ePODFileExpOpt,
	ePODFileHistory,
	ePODFileEndiannessMisMatch  = -402456576,

	ePODFileColourBackground	= 2000,
	ePODFileColourAmbient,
	ePODFileNumCamera,
	ePODFileNumLight,
	ePODFileNumMesh,
	ePODFileNumNode,
	ePODFileNumMeshNode,
	ePODFileNumTexture,
	ePODFileNumMaterial,
	ePODFileNumFrame,
	ePODFileCamera,		// Will come multiple times
	ePODFileLight,		// Will come multiple times
	ePODFileMesh,		// Will come multiple times
	ePODFileNode,		// Will come multiple times
	ePODFileTexture,	// Will come multiple times
	ePODFileMaterial,	// Will come multiple times
	ePODFileFlags,

	ePODFileMatName				= 3000,
	ePODFileMatIdxTexDiffuse,
	ePODFileMatOpacity,
	ePODFileMatAmbient,
	ePODFileMatDiffuse,
	ePODFileMatSpecular,
	ePODFileMatShininess,
	ePODFileMatEffectFile,
	ePODFileMatEffectName,
	ePODFileMatIdxTexAmbient,
	ePODFileMatIdxTexSpecularColour,
	ePODFileMatIdxTexSpecularLevel,
	ePODFileMatIdxTexBump,
	ePODFileMatIdxTexEmissive,
	ePODFileMatIdxTexGlossiness,
	ePODFileMatIdxTexOpacity,
	ePODFileMatIdxTexReflection,
	ePODFileMatIdxTexRefraction,
	ePODFileMatBlendSrcRGB,
	ePODFileMatBlendSrcA,
	ePODFileMatBlendDstRGB,
	ePODFileMatBlendDstA,
	ePODFileMatBlendOpRGB,
	ePODFileMatBlendOpA,
	ePODFileMatBlendColour,
	ePODFileMatBlendFactor,
	ePODFileMatFlags,

	ePODFileTexName				= 4000,

	ePODFileNodeIdx				= 5000,
	ePODFileNodeName,
	ePODFileNodeIdxMat,
	ePODFileNodeIdxParent,
	ePODFileNodePos,
	ePODFileNodeRot,
	ePODFileNodeScale,
	ePODFileNodeAnimPos,
	ePODFileNodeAnimRot,
	ePODFileNodeAnimScale,
	ePODFileNodeMatrix,
	ePODFileNodeAnimMatrix,
	ePODFileNodeAnimFlags,

	ePODFileMeshNumVtx			= 6000,
	ePODFileMeshNumFaces,
	ePODFileMeshNumUVW,
	ePODFileMeshFaces,
	ePODFileMeshStripLength,
	ePODFileMeshNumStrips,
	ePODFileMeshVtx,
	ePODFileMeshNor,
	ePODFileMeshTan,
	ePODFileMeshBin,
	ePODFileMeshUVW,			// Will come multiple times
	ePODFileMeshVtxCol,
	ePODFileMeshBoneIdx,
	ePODFileMeshBoneWeight,
	ePODFileMeshInterleaved,
	ePODFileMeshBoneBatches,
	ePODFileMeshBoneBatchBoneCnts,
	ePODFileMeshBoneBatchOffsets,
	ePODFileMeshBoneBatchBoneMax,
	ePODFileMeshBoneBatchCnt,

	ePODFileLightIdxTgt			= 7000,
	ePODFileLightColour,
	ePODFileLightType,
	ePODFileLightConstantAttenuation,
	ePODFileLightLinearAttenuation,
	ePODFileLightQuadraticAttenuation,
	ePODFileLightFalloffAngle,
	ePODFileLightFalloffExponent,

	ePODFileCamIdxTgt			= 8000,
	ePODFileCamFOV,
	ePODFileCamFar,
	ePODFileCamNear,
	ePODFileCamAnimFOV,

	ePODFileDataType			= 9000,
	ePODFileN,
	ePODFileStride,
	ePODFileData
};

/****************************************************************************
** Structures
****************************************************************************/
struct SPVRTPODImpl
{
	VERTTYPE	fFrame;		/*!< Frame number */
	VERTTYPE	fBlend;		/*!< Frame blend	(AKA fractional part of animation frame number) */
	int			nFrame;		/*!< Frame number (AKA integer part of animation frame number) */

	VERTTYPE	*pfCache;		/*!< Cache indicating the frames at which the matrix cache was filled */
	PVRTMATRIX	*pWmCache;		/*!< Cache of world matrices */
	PVRTMATRIX	*pWmZeroCache;	/*!< Pre-calculated frame 0 matrices */

	bool		bFromMemory;	/*!< Was the mesh data loaded from memory? */

#ifdef _DEBUG
	__int64	nWmTotal, nWmCacheHit, nWmZeroCacheHit;
	float	fHitPerc, fHitPercZero;
#endif
};

/****************************************************************************
** Local code: Memory allocation
****************************************************************************/

/*!***************************************************************************
 @Function			SafeAlloc
 @Input				cnt
 @Output			ptr
 @Return			false if memory allocation failed
 @Description		Allocates a block of memory.
*****************************************************************************/
template <typename T>
bool SafeAlloc(T* &ptr, size_t cnt)
{
	_ASSERT(!ptr);
	if(cnt)
	{
		ptr = (T*)calloc(cnt, sizeof(T));
		_ASSERT(ptr);
		if(!ptr)
			return false;
	}
	return true;
}

#ifdef _UITRON_
template bool SafeAlloc<unsigned int>(unsigned int*&,size_t);
template bool SafeAlloc<SPODTexture>(SPODTexture*&,size_t);
template bool SafeAlloc<SPODLight>(SPODLight*&,size_t);
template bool SafeAlloc<SPODNode>(SPODNode*&,size_t);
template bool SafeAlloc<unsigned char>(unsigned char*&,size_t);
template bool SafeAlloc<int>(int*&,size_t);
template bool SafeAlloc<float>(float*&,size_t);
template bool SafeAlloc<CPODData>(CPODData*&,size_t);
template bool SafeAlloc<char>(char*&,size_t);
template bool SafeAlloc<SPODCamera>(SPODCamera*&,size_t);
template bool SafeAlloc<SPODMesh>(SPODMesh*&,size_t);
template bool SafeAlloc<SPODMaterial>(SPODMaterial*&,size_t);
#endif


/*!***************************************************************************
 @Function			SafeRealloc
 @Modified			ptr
 @Input				cnt
 @Description		Changes the size of a memory allocation.
*****************************************************************************/
template <typename T>
void SafeRealloc(T* &ptr, size_t cnt)
{
	ptr = (T*)realloc(ptr, cnt * sizeof(T));
	_ASSERT(ptr);
}

#ifdef _UITRON_
template void SafeRealloc<unsigned char>(unsigned char*&,size_t);
#endif

/****************************************************************************
** Class: CPODData
****************************************************************************/
/*!***************************************************************************
@Function			Reset
@Description		Resets the POD Data to NULL
*****************************************************************************/
void CPODData::Reset()
{
	eType = EPODDataNone;
	n = 0;
	nStride = 0;
	FREE(pData);
}

/*!***************************************************************************
 Class: CSource
*****************************************************************************/
class CSource
{
public:
	/*!***************************************************************************
	@Function			~CSource
	@Description		Destructor
	*****************************************************************************/
	virtual ~CSource() {};
	virtual bool Read(void* lpBuffer, const unsigned int dwNumberOfBytesToRead) = 0;
	virtual bool Skip(const unsigned int nBytes) = 0;

	template <typename T>
	bool Read(T &n)
	{
		return Read(&n, sizeof(T));
	}

	template <typename T>
	bool Read32(T &n)
	{
		unsigned char ub[4];

		if(Read(&ub, 4))
		{
			unsigned int *pn = (unsigned int*) &n;
			*pn = (unsigned int) ((ub[3] << 24) | (ub[2] << 16) | (ub[1] << 8) | ub[0]);
			return true;
		}

		return false;
	}

	template <typename T>
	bool Read16(T &n)
	{
		unsigned char ub[2];

		if(Read(&ub, 2))
		{
			unsigned short *pn = (unsigned short*) &n;
			*pn = (unsigned short) ((ub[1] << 8) | ub[0]);
			return true;
		}

		return false;
	}

	bool ReadMarker(unsigned int &nName, unsigned int &nLen);

	template <typename T>
	bool ReadAfterAlloc(T* &lpBuffer, const unsigned int dwNumberOfBytesToRead)
	{
		if(!SafeAlloc(lpBuffer, dwNumberOfBytesToRead))
			return false;
		return Read(lpBuffer, dwNumberOfBytesToRead);
	}

	template <typename T>
	bool ReadAfterAlloc32(T* &lpBuffer, const unsigned int dwNumberOfBytesToRead)
	{
		if(!SafeAlloc(lpBuffer, dwNumberOfBytesToRead))
			return false;
		return ReadArray32((unsigned int*) lpBuffer, dwNumberOfBytesToRead / 4);
	}

	bool ReadArray32(unsigned int *pn, unsigned int i32Size)
	{
		bool bRet = true;

		for(unsigned int i = 0; i < i32Size; ++i)
			bRet &= Read32(pn[i]);

		return bRet;
	}

	template <typename T>
	bool ReadAfterAlloc16(T* &lpBuffer, const unsigned int dwNumberOfBytesToRead)
	{
		if(!SafeAlloc(lpBuffer, dwNumberOfBytesToRead))
			return false;
		return ReadArray16((unsigned short*) lpBuffer, dwNumberOfBytesToRead / 2);
	}

	bool ReadArray16(unsigned short* pn, unsigned int i32Size)
	{
		bool bRet = true;

		for(unsigned int i = 0; i < i32Size; ++i)
			bRet &= Read16(pn[i]);

		return bRet;
	}
};

bool CSource::ReadMarker(unsigned int &nName, unsigned int &nLen)
{
	if(!Read32(nName))
		return false;
	if(!Read32(nLen))
		return false;
	return true;
}

/*!***************************************************************************
 Class: CSourceStream
*****************************************************************************/
class CSourceStream : public CSource
{
protected:
	CPVRTResourceFile* m_pFile;
	size_t m_BytesReadCount;

public:
	/*!***************************************************************************
	@Function			CSourceStream
	@Description		Constructor
	*****************************************************************************/
	CSourceStream() : m_pFile(0), m_BytesReadCount(0) {}

	/*!***************************************************************************
	@Function			~CSourceStream
	@Description		Destructor
	*****************************************************************************/
	virtual ~CSourceStream();

	bool Init(const char * const pszFileName);
	bool Init(const char * const pData, const size_t i32Size);

	virtual bool Read(void* lpBuffer, const unsigned int dwNumberOfBytesToRead);
	virtual bool Skip(const unsigned int nBytes);
};

/*!***************************************************************************
@Function			~CSourceStream
@Description		Destructor
*****************************************************************************/
CSourceStream::~CSourceStream()
{
	delete m_pFile;
}

bool CSourceStream::Init(const char * const pszFileName)
{
	m_BytesReadCount = 0;
	if (m_pFile) delete m_pFile;

	m_pFile = new CPVRTResourceFile(pszFileName);
	if (!m_pFile->IsOpen())
	{
		delete m_pFile;
		m_pFile = 0;
		return false;
	}
	return true;
}

bool CSourceStream::Init(const char * pData, size_t i32Size)
{
	m_BytesReadCount = 0;
	if (m_pFile) delete m_pFile;

	m_pFile = new CPVRTResourceFile(pData, i32Size);
	if (!m_pFile->IsOpen())
	{
		delete m_pFile;
		m_pFile = 0;
		return false;
	}
	return true;
}

bool CSourceStream::Read(void* lpBuffer, const unsigned int dwNumberOfBytesToRead)
{
	_ASSERT(lpBuffer);
	_ASSERT(m_pFile);

	if (m_BytesReadCount + dwNumberOfBytesToRead > m_pFile->Size()) return false;

	memcpy(lpBuffer, &(m_pFile->StringPtr())[m_BytesReadCount], dwNumberOfBytesToRead);

	m_BytesReadCount += dwNumberOfBytesToRead;
	return true;
}

bool CSourceStream::Skip(const unsigned int nBytes)
{
	if (m_BytesReadCount + nBytes > m_pFile->Size()) return false;
	m_BytesReadCount += nBytes;
	return true;
}

#ifdef WIN32
/*!***************************************************************************
 Class: CSourceResource
*****************************************************************************/
class CSourceResource : public CSource
{
protected:
	const unsigned char	*m_pData;
	unsigned int		m_nSize, m_nReadPos;

public:
	bool Init(const TCHAR * const pszName);
	virtual bool Read(void* lpBuffer, const unsigned int dwNumberOfBytesToRead);
	virtual bool Skip(const unsigned int nBytes);
};

bool CSourceResource::Init(const TCHAR * const pszName)
{
	HRSRC	hR;
	HGLOBAL	hG;

	// Find the resource
	hR = FindResource(GetModuleHandle(NULL), pszName, RT_RCDATA);
	if(!hR)
		return false;

	// How big is the resource?
	m_nSize = SizeofResource(NULL, hR);
	if(!m_nSize)
		return false;

	// Get a pointer to the resource data
	hG = LoadResource(NULL, hR);
	if(!hG)
		return false;

	m_pData = (unsigned char*)LockResource(hG);
	if(!m_pData)
		return false;

	m_nReadPos = 0;
	return true;
}

bool CSourceResource::Read(void* lpBuffer, const unsigned int dwNumberOfBytesToRead)
{
	if(m_nReadPos + dwNumberOfBytesToRead > m_nSize)
		return false;

	_ASSERT(lpBuffer);
	memcpy(lpBuffer, &m_pData[m_nReadPos], dwNumberOfBytesToRead);
	m_nReadPos += dwNumberOfBytesToRead;
	return true;
}

bool CSourceResource::Skip(const unsigned int nBytes)
{
	if(m_nReadPos + nBytes > m_nSize)
		return false;

	m_nReadPos += nBytes;
	return true;
}

#endif /* WIN32 */

/****************************************************************************
** Local code: File writing
****************************************************************************/

/*!***************************************************************************
 @Function			WriteFileSafe
 @Input				pFile
 @Input				lpBuffer
 @Input				nNumberOfBytesToWrite
 @Return			true if successful
 @Description		Writes data to a file, checking return codes.
*****************************************************************************/
static bool WriteFileSafe(FILE *pFile, const void * const lpBuffer, const unsigned int nNumberOfBytesToWrite)
{
	if(nNumberOfBytesToWrite)
	{
		size_t count = fwrite(lpBuffer, nNumberOfBytesToWrite, 1, pFile);
		return count == 1;
	}
	return true;
}

static bool WriteFileSafe16(FILE *pFile, const unsigned short * const lpBuffer, const unsigned int nSize)
{
	if(nSize)
	{
		unsigned char ub[2];
		bool bRet = true;

		for(unsigned int i = 0; i < nSize; ++i)
		{
			ub[0] = (unsigned char) lpBuffer[i];
			ub[1] = lpBuffer[i] >> 8;

			bRet &= (fwrite(ub, 2, 1, pFile) == 1);
		}

		return bRet;
	}
	return true;
}

static bool WriteFileSafe32(FILE *pFile, const unsigned int * const lpBuffer, const unsigned int nSize)
{
	if(nSize)
	{
		unsigned char ub[4];
		bool bRet = true;

		for(unsigned int i = 0; i < nSize; ++i)
		{
			ub[0] = lpBuffer[i];
			ub[1] = lpBuffer[i] >> 8;
			ub[2] = lpBuffer[i] >> 16;
			ub[3] = lpBuffer[i] >> 24;

			bRet &= (fwrite(ub, 4, 1, pFile) == 1);
		}

		return bRet;
	}
	return true;
}
/*!***************************************************************************
 @Function			WriteMarker
 @Input				pFile
 @Input				nName
 @Input				bEnd
 @Input				nLen
 Return				true if successful
 @Description		Write a marker to a POD file. If bEnd if false, it's a
					beginning marker, otherwise it's an end marker.
*****************************************************************************/
static bool WriteMarker(
	FILE				* const pFile,
	const unsigned int	nName,
	const bool			bEnd,
	const unsigned int	nLen = 0)
{
	unsigned int nMarker;
	bool bRet;

	_ASSERT((nName & ~PVRTMODELPOD_TAG_MASK) == nName);
	nMarker = nName | (bEnd ? PVRTMODELPOD_TAG_END : PVRTMODELPOD_TAG_START);

	bRet  = WriteFileSafe32(pFile, &nMarker, 1);
	bRet &= WriteFileSafe32(pFile, &nLen, 1);

	return bRet;
}

/*!***************************************************************************
 @Function			WriteData
 @Input				pFile
 @Input				nName
 @Input				pData
 @Input				nLen
 @Return			true if successful
 @Description		Write nLen bytes of data from pData, bracketed by an nName
					begin/end markers.
*****************************************************************************/
static bool WriteData(
	FILE				* const pFile,
	const unsigned int	nName,
	const void			* const pData,
	const unsigned int	nLen)
{
	if(pData)
	{
		_ASSERT(nLen);
		if(!WriteMarker(pFile, nName, false, nLen)) return false;
		if(!WriteFileSafe(pFile, pData, nLen)) return false;
		if(!WriteMarker(pFile, nName, true)) return false;
	}
	return true;
}

/*!***************************************************************************
 @Function			WriteData16
 @Input				pFile
 @Input				nName
 @Input				pData
 @Input				i32Size
 @Return			true if successful
 @Description		Write i32Size no. of unsigned shorts from pData, bracketed by
					an nName begin/end markers.
*****************************************************************************/
template <typename T>
static bool WriteData16(
	FILE				* const pFile,
	const unsigned int	nName,
	const T	* const pData,
	int i32Size = 1)
{
	if(pData)
	{
		if(!WriteMarker(pFile, nName, false, 2 * i32Size)) return false;
		if(!WriteFileSafe16(pFile, (unsigned short*) pData, i32Size)) return false;
		if(!WriteMarker(pFile, nName, true)) return false;
	}
	return true;
}

/*!***************************************************************************
 @Function			WriteData32
 @Input				pFile
 @Input				nName
 @Input				pData
 @Input				i32Size
 @Return			true if successful
 @Description		Write i32Size no. of unsigned ints from pData, bracketed by
					an nName begin/end markers.
*****************************************************************************/
template <typename T>
static bool WriteData32(
	FILE				* const pFile,
	const unsigned int	nName,
	const T	* const pData,
	int i32Size = 1)
{
	if(pData)
	{
		if(!WriteMarker(pFile, nName, false, 4 * i32Size)) return false;
		if(!WriteFileSafe32(pFile, (unsigned int*) pData, i32Size)) return false;
		if(!WriteMarker(pFile, nName, true)) return false;
	}
	return true;
}

/*!***************************************************************************
 @Function			WriteData
 @Input				pFile
 @Input				nName
 @Input				n
 @Return			true if successful
 @Description		Write the value n, bracketed by an nName begin/end markers.
*****************************************************************************/
template <typename T>
static bool WriteData(
	FILE				* const pFile,
	const unsigned int	nName,
	const T				&n)
{
	unsigned int nSize = sizeof(T);

	bool bRet = WriteData(pFile, nName, (void*)&n, nSize);

	return bRet;
}

/*!***************************************************************************
 @Function			WriteCPODData
 @Input				pFile
 @Input				nName
 @Input				n
 @Input				nEntries
 @Input				bValidData
 @Return			true if successful
 @Description		Write the value n, bracketed by an nName begin/end markers.
*****************************************************************************/
static bool WriteCPODData(
	FILE				* const pFile,
	const unsigned int	nName,
	const CPODData		&n,
	const unsigned int	nEntries,
	const bool			bValidData)
{
	if(!WriteMarker(pFile, nName, false)) return false;
	if(!WriteData32(pFile, ePODFileDataType, &n.eType)) return false;
	if(!WriteData32(pFile, ePODFileN, &n.n)) return false;
	if(!WriteData32(pFile, ePODFileStride, &n.nStride)) return false;
	if(bValidData)
	{
		switch(PVRTModelPODDataTypeSize(n.eType))
		{
			case 1: if(!WriteData(pFile, ePODFileData, n.pData, nEntries * n.nStride)) return false; break;
			case 2: if(!WriteData16(pFile, ePODFileData, n.pData, nEntries * (n.nStride / 2))) return false; break;
			case 4: if(!WriteData32(pFile, ePODFileData, n.pData, nEntries * (n.nStride / 4))) return false; break;
			default: { _ASSERT(false); }
		};
	}
	else
	{
		unsigned int offset = (unsigned int) n.pData;
		if(!WriteData32(pFile, ePODFileData, &offset)) return false;
	}
	if(!WriteMarker(pFile, nName, true)) return false;
	return true;
}

/*!***************************************************************************
 @Function			WriteSingleValueIntoInterleaved
 @Input				pFile
 @Input				pData
 @Input				nNo
 @Input				nBytes
 @Input				nStride
 @Return			true if successful
 @Description		Write a single value (e.g. position, normal) from pData
					to the file.
*****************************************************************************/
static bool WriteSingleValueIntoInterleaved(FILE * const pFile, unsigned char **pData, unsigned int nNo, size_t nBytes, unsigned int nStride)
{
	if(*pData)
	{
		switch(nBytes)
		{
			case 1: if(!WriteFileSafe(pFile, *pData, nNo)) return false; break;
			case 2: if(!WriteFileSafe16(pFile, (unsigned short*) *pData, nNo)) return false; break;
			case 4: if(!WriteFileSafe32(pFile, (unsigned int*) *pData, nNo)) return false; break;
			default: { _ASSERT(false); }
		};

		*pData += nStride;
	}

	return true;
}

/*!***************************************************************************
 @Function			WriteInterleaved
 @Input				pFile
 @Input				mesh
 @Return			true if successful
 @Description		Write out the interleaved data to file. Always assumes the
					interleaved data is in a particular order.
*****************************************************************************/
static bool WriteInterleaved(FILE * const pFile, SPODMesh &mesh)
{
	if(!mesh.pInterleaved)
		return true;

	unsigned int i;
	unsigned char * pVertices, *pNormals, *pTangents, *pBinormals, *pVColours, **pUVW = 0, *pBoneID, *pBoneWeight;

	pVertices = mesh.sVertex.n ? mesh.pInterleaved + (size_t) mesh.sVertex.pData  : 0;
	pNormals  = mesh.sNormals.n ? mesh.pInterleaved + (size_t) mesh.sNormals.pData : 0;
	pTangents = mesh.sTangents.n ? mesh.pInterleaved + (size_t) mesh.sTangents.pData : 0;
	pBinormals= mesh.sBinormals.n ? mesh.pInterleaved + (size_t) mesh.sBinormals.pData : 0;
	pVColours = mesh.sVtxColours.n ? mesh.pInterleaved + (size_t) mesh.sVtxColours.pData : 0;

	if(mesh.nNumUVW)
	{
		pUVW = new unsigned char*[mesh.nNumUVW];

		for(i = 0; i < mesh.nNumUVW; ++i)
			pUVW[i] = mesh.psUVW[i].n ? mesh.pInterleaved + (size_t) mesh.psUVW[i].pData : 0;
	}

	pBoneID = mesh.sBoneIdx.n ? mesh.pInterleaved + (size_t) mesh.sBoneIdx.pData : 0;
	pBoneWeight = mesh.sBoneWeight.n ? mesh.pInterleaved + (size_t) mesh.sBoneWeight.pData : 0;

	if(!WriteMarker(pFile, ePODFileMeshInterleaved, false, mesh.nNumVertex * mesh.sVertex.nStride)) return false;

	for(i = 0; i < mesh.nNumVertex; ++i)
	{
		WriteSingleValueIntoInterleaved(pFile, &pVertices, mesh.sVertex.n, PVRTModelPODDataTypeSize(mesh.sVertex.eType), mesh.sVertex.nStride);
		WriteSingleValueIntoInterleaved(pFile, &pNormals, mesh.sNormals.n, PVRTModelPODDataTypeSize(mesh.sNormals.eType), mesh.sNormals.nStride);
		WriteSingleValueIntoInterleaved(pFile, &pTangents, mesh.sTangents.n, PVRTModelPODDataTypeSize(mesh.sTangents.eType), mesh.sTangents.nStride);
		WriteSingleValueIntoInterleaved(pFile, &pBinormals, mesh.sBinormals.n, PVRTModelPODDataTypeSize(mesh.sBinormals.eType), mesh.sBinormals.nStride);
		WriteSingleValueIntoInterleaved(pFile, &pVColours, mesh.sVtxColours.n, PVRTModelPODDataTypeSize(mesh.sVtxColours.eType), mesh.sVtxColours.nStride);

		for(unsigned int j = 0; j < mesh.nNumUVW; ++j)
			WriteSingleValueIntoInterleaved(pFile, &pUVW[j], mesh.psUVW[j].n, PVRTModelPODDataTypeSize(mesh.psUVW[j].eType), mesh.psUVW[j].nStride);

		WriteSingleValueIntoInterleaved(pFile, &pBoneID, mesh.sBoneIdx.n, PVRTModelPODDataTypeSize(mesh.sBoneIdx.eType), mesh.sBoneIdx.nStride);
		WriteSingleValueIntoInterleaved(pFile, &pBoneWeight, mesh.sBoneWeight.n, PVRTModelPODDataTypeSize(mesh.sBoneWeight.eType), mesh.sBoneWeight.nStride);
	}

	delete[] pUVW;

	if(!WriteMarker(pFile, ePODFileMeshInterleaved, true)) return false;
	return true;
}

/*!***************************************************************************
 @Function			WritePOD
 @Output			The file referenced by pFile
 @Input				s The POD Scene to write
 @Input				pszExpOpt Exporter options
 @Return			true if successful
 @Description		Write a POD file
*****************************************************************************/
static bool WritePOD(
	FILE			* const pFile,
	const char		* const pszExpOpt,
	const char		* const pszHistory,
	const SPODScene	&s)
{
	unsigned int i, j;

	// Save: file version
	{
		char *pszVersion = (char*)PVRTMODELPOD_VERSION;

		if(!WriteData(pFile, ePODFileVersion, pszVersion, (unsigned int)strlen(pszVersion) + 1)) return false;
	}

	// Save: exporter options
	if(pszExpOpt && *pszExpOpt)
	{
		if(!WriteData(pFile, ePODFileExpOpt, pszExpOpt, (unsigned int)strlen(pszExpOpt) + 1)) return false;
	}

	// Save: .pod file history
	if(pszHistory && *pszHistory)
	{
		if(!WriteData(pFile, ePODFileHistory, pszHistory, (unsigned int)strlen(pszHistory) + 1)) return false;
	}

	// Save: scene descriptor
	if(!WriteMarker(pFile, ePODFileScene, false)) return false;

	{
		if(!WriteData32(pFile, ePODFileColourBackground,	s.pfColourBackground, sizeof(s.pfColourBackground) / sizeof(*s.pfColourBackground))) return false;
		if(!WriteData32(pFile, ePODFileColourAmbient,		s.pfColourAmbient, sizeof(s.pfColourAmbient) / sizeof(*s.pfColourAmbient))) return false;
		if(!WriteData32(pFile, ePODFileNumCamera, &s.nNumCamera)) return false;
		if(!WriteData32(pFile, ePODFileNumLight, &s.nNumLight)) return false;
		if(!WriteData32(pFile, ePODFileNumMesh,	&s.nNumMesh)) return false;
		if(!WriteData32(pFile, ePODFileNumNode,	&s.nNumNode)) return false;
		if(!WriteData32(pFile, ePODFileNumMeshNode,	&s.nNumMeshNode)) return false;
		if(!WriteData32(pFile, ePODFileNumTexture, &s.nNumTexture)) return false;
		if(!WriteData32(pFile, ePODFileNumMaterial,	&s.nNumMaterial)) return false;
		if(!WriteData32(pFile, ePODFileNumFrame, &s.nNumFrame)) return false;
		if(!WriteData32(pFile, ePODFileFlags, &s.nFlags)) return false;
		// Save: cameras
		for(i = 0; i < s.nNumCamera; ++i)
		{
			if(!WriteMarker(pFile, ePODFileCamera, false)) return false;
			if(!WriteData32(pFile, ePODFileCamIdxTgt, &s.pCamera[i].nIdxTarget)) return false;
			if(!WriteData32(pFile, ePODFileCamFOV,	  &s.pCamera[i].fFOV)) return false;
			if(!WriteData32(pFile, ePODFileCamFar,	  &s.pCamera[i].fFar)) return false;
			if(!WriteData32(pFile, ePODFileCamNear,	  &s.pCamera[i].fNear)) return false;
			if(!WriteData32(pFile, ePODFileCamAnimFOV,	s.pCamera[i].pfAnimFOV, s.nNumFrame)) return false;
			if(!WriteMarker(pFile, ePODFileCamera, true)) return false;
		}
		// Save: lights
		for(i = 0; i < s.nNumLight; ++i)
		{
			if(!WriteMarker(pFile, ePODFileLight, false)) return false;
			if(!WriteData32(pFile, ePODFileLightIdxTgt,	&s.pLight[i].nIdxTarget)) return false;
			if(!WriteData32(pFile, ePODFileLightColour,	s.pLight[i].pfColour, sizeof(s.pLight[i].pfColour) / sizeof(*s.pLight[i].pfColour))) return false;
			if(!WriteData32(pFile, ePODFileLightType,	&s.pLight[i].eType)) return false;

			if(s.pLight[i].eType != ePODDirectional)
			{
				if(!WriteData32(pFile, ePODFileLightConstantAttenuation,	&s.pLight[i].fConstantAttenuation))  return false;
				if(!WriteData32(pFile, ePODFileLightLinearAttenuation,		&s.pLight[i].fLinearAttenuation))	  return false;
				if(!WriteData32(pFile, ePODFileLightQuadraticAttenuation,	&s.pLight[i].fQuadraticAttenuation)) return false;
			}

			if(s.pLight[i].eType == ePODSpot)
			{
				if(!WriteData32(pFile, ePODFileLightFalloffAngle,			&s.pLight[i].fFalloffAngle))		  return false;
				if(!WriteData32(pFile, ePODFileLightFalloffExponent,		&s.pLight[i].fFalloffExponent))	  return false;
			}

			if(!WriteMarker(pFile, ePODFileLight, true)) return false;
		}

		// Save: materials
		for(i = 0; i < s.nNumMaterial; ++i)
		{
			if(!WriteMarker(pFile, ePODFileMaterial, false)) return false;
			if(!WriteData32(pFile, ePODFileMatFlags,  &s.pMaterial[i].nFlags)) return false;
			if(!WriteData(pFile,   ePODFileMatName,			s.pMaterial[i].pszName, (unsigned int)strlen(s.pMaterial[i].pszName)+1)) return false;
			if(!WriteData32(pFile, ePODFileMatIdxTexDiffuse,	&s.pMaterial[i].nIdxTexDiffuse)) return false;
			if(!WriteData32(pFile, ePODFileMatIdxTexAmbient,	&s.pMaterial[i].nIdxTexAmbient)) return false;
			if(!WriteData32(pFile, ePODFileMatIdxTexSpecularColour,	&s.pMaterial[i].nIdxTexSpecularColour)) return false;
			if(!WriteData32(pFile, ePODFileMatIdxTexSpecularLevel,	&s.pMaterial[i].nIdxTexSpecularLevel)) return false;
			if(!WriteData32(pFile, ePODFileMatIdxTexBump,	&s.pMaterial[i].nIdxTexBump)) return false;
			if(!WriteData32(pFile, ePODFileMatIdxTexEmissive,	&s.pMaterial[i].nIdxTexEmissive)) return false;
			if(!WriteData32(pFile, ePODFileMatIdxTexGlossiness,	&s.pMaterial[i].nIdxTexGlossiness)) return false;
			if(!WriteData32(pFile, ePODFileMatIdxTexOpacity,	&s.pMaterial[i].nIdxTexOpacity)) return false;
			if(!WriteData32(pFile, ePODFileMatIdxTexReflection,	&s.pMaterial[i].nIdxTexReflection)) return false;
			if(!WriteData32(pFile, ePODFileMatIdxTexRefraction,	&s.pMaterial[i].nIdxTexRefraction)) return false;
			if(!WriteData32(pFile, ePODFileMatOpacity,	&s.pMaterial[i].fMatOpacity)) return false;
			if(!WriteData32(pFile, ePODFileMatAmbient,		s.pMaterial[i].pfMatAmbient, sizeof(s.pMaterial[i].pfMatAmbient) / sizeof(*s.pMaterial[i].pfMatAmbient))) return false;
			if(!WriteData32(pFile, ePODFileMatDiffuse,		s.pMaterial[i].pfMatDiffuse, sizeof(s.pMaterial[i].pfMatDiffuse) / sizeof(*s.pMaterial[i].pfMatDiffuse))) return false;
			if(!WriteData32(pFile, ePODFileMatSpecular,		s.pMaterial[i].pfMatSpecular, sizeof(s.pMaterial[i].pfMatSpecular) / sizeof(*s.pMaterial[i].pfMatSpecular))) return false;
			if(!WriteData32(pFile, ePODFileMatShininess, &s.pMaterial[i].fMatShininess)) return false;
			if(!WriteData(pFile, ePODFileMatEffectFile,		s.pMaterial[i].pszEffectFile, s.pMaterial[i].pszEffectFile ? ((unsigned int)strlen(s.pMaterial[i].pszEffectFile)+1) : 0)) return false;
			if(!WriteData(pFile, ePODFileMatEffectName,		s.pMaterial[i].pszEffectName, s.pMaterial[i].pszEffectName ? ((unsigned int)strlen(s.pMaterial[i].pszEffectName)+1) : 0)) return false;
			if(!WriteData32(pFile, ePODFileMatBlendSrcRGB,  &s.pMaterial[i].eBlendSrcRGB))return false;
			if(!WriteData32(pFile, ePODFileMatBlendSrcA,	&s.pMaterial[i].eBlendSrcA))	return false;
			if(!WriteData32(pFile, ePODFileMatBlendDstRGB,  &s.pMaterial[i].eBlendDstRGB))return false;
			if(!WriteData32(pFile, ePODFileMatBlendDstA,	&s.pMaterial[i].eBlendDstA))	return false;
			if(!WriteData32(pFile, ePODFileMatBlendOpRGB,	&s.pMaterial[i].eBlendOpRGB)) return false;
			if(!WriteData32(pFile, ePODFileMatBlendOpA,		&s.pMaterial[i].eBlendOpA))	return false;
			if(!WriteData32(pFile, ePODFileMatBlendColour, s.pMaterial[i].pfBlendColour, sizeof(s.pMaterial[i].pfBlendColour) / sizeof(*s.pMaterial[i].pfBlendColour))) return false;
			if(!WriteData32(pFile, ePODFileMatBlendFactor, s.pMaterial[i].pfBlendFactor, sizeof(s.pMaterial[i].pfBlendFactor) / sizeof(*s.pMaterial[i].pfBlendFactor))) return false;
			if(!WriteMarker(pFile, ePODFileMaterial, true)) return false;
		}

		// Save: meshes
		for(i = 0; i < s.nNumMesh; ++i)
		{
			if(!WriteMarker(pFile, ePODFileMesh, false)) return false;

			if(!WriteData32(pFile, ePODFileMeshNumVtx,			&s.pMesh[i].nNumVertex)) return false;
			if(!WriteData32(pFile, ePODFileMeshNumFaces,		&s.pMesh[i].nNumFaces)) return false;
			if(!WriteData32(pFile, ePODFileMeshNumUVW,			&s.pMesh[i].nNumUVW)) return false;
			if(!WriteData32(pFile, ePODFileMeshStripLength,		s.pMesh[i].pnStripLength, s.pMesh[i].nNumStrips)) return false;
			if(!WriteData32(pFile, ePODFileMeshNumStrips,		&s.pMesh[i].nNumStrips)) return false;
			if(!WriteInterleaved(pFile, s.pMesh[i])) return false;
			if(!WriteData32(pFile, ePODFileMeshBoneBatchBoneMax,&s.pMesh[i].sBoneBatches.nBatchBoneMax)) return false;
			if(!WriteData32(pFile, ePODFileMeshBoneBatchCnt,	&s.pMesh[i].sBoneBatches.nBatchCnt)) return false;
			if(!WriteData32(pFile, ePODFileMeshBoneBatches,		s.pMesh[i].sBoneBatches.pnBatches, s.pMesh[i].sBoneBatches.nBatchBoneMax * s.pMesh[i].sBoneBatches.nBatchCnt)) return false;
			if(!WriteData32(pFile, ePODFileMeshBoneBatchBoneCnts,	s.pMesh[i].sBoneBatches.pnBatchBoneCnt, s.pMesh[i].sBoneBatches.nBatchCnt)) return false;
			if(!WriteData32(pFile, ePODFileMeshBoneBatchOffsets,	s.pMesh[i].sBoneBatches.pnBatchOffset,s.pMesh[i].sBoneBatches.nBatchCnt)) return false;

			if(!WriteCPODData(pFile, ePODFileMeshFaces,			s.pMesh[i].sFaces,		PVRTModelPODCountIndices(s.pMesh[i]), true)) return false;
			if(!WriteCPODData(pFile, ePODFileMeshVtx,			s.pMesh[i].sVertex,		s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0)) return false;
			if(!WriteCPODData(pFile, ePODFileMeshNor,			s.pMesh[i].sNormals,	s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0)) return false;
			if(!WriteCPODData(pFile, ePODFileMeshTan,			s.pMesh[i].sTangents,	s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0)) return false;
			if(!WriteCPODData(pFile, ePODFileMeshBin,			 s.pMesh[i].sBinormals,	s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0)) return false;

			for(j = 0; j < s.pMesh[i].nNumUVW; ++j)
				if(!WriteCPODData(pFile, ePODFileMeshUVW,		s.pMesh[i].psUVW[j],	s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0)) return false;

			if(!WriteCPODData(pFile, ePODFileMeshVtxCol,		s.pMesh[i].sVtxColours, s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0)) return false;
			if(!WriteCPODData(pFile, ePODFileMeshBoneIdx,		s.pMesh[i].sBoneIdx,	s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0)) return false;
			if(!WriteCPODData(pFile, ePODFileMeshBoneWeight,	s.pMesh[i].sBoneWeight,	s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0)) return false;

			if(!WriteMarker(pFile, ePODFileMesh, true)) return false;
		}

		int iTransformationNo;
		// Save: node
		for(i = 0; i < s.nNumNode; ++i)
		{
			if(!WriteMarker(pFile, ePODFileNode, false)) return false;

			{
				if(!WriteData32(pFile, ePODFileNodeIdx,		&s.pNode[i].nIdx)) return false;
				if(!WriteData(pFile, ePODFileNodeName,		s.pNode[i].pszName, (unsigned int)strlen(s.pNode[i].pszName)+1)) return false;
				if(!WriteData32(pFile, ePODFileNodeIdxMat,	&s.pNode[i].nIdxMaterial)) return false;
				if(!WriteData32(pFile, ePODFileNodeIdxParent, &s.pNode[i].nIdxParent)) return false;
				if(!WriteData32(pFile, ePODFileNodeAnimFlags, &s.pNode[i].nAnimFlags)) return false;

				iTransformationNo = s.pNode[i].nAnimFlags & ePODHasPositionAni ? s.nNumFrame : 1;
				if(!WriteData32(pFile, ePODFileNodeAnimPos,	s.pNode[i].pfAnimPosition,	iTransformationNo * 3)) return false;

				iTransformationNo = s.pNode[i].nAnimFlags & ePODHasRotationAni ? s.nNumFrame : 1;
				if(!WriteData32(pFile, ePODFileNodeAnimRot,	s.pNode[i].pfAnimRotation,	iTransformationNo * 4)) return false;

				iTransformationNo = s.pNode[i].nAnimFlags & ePODHasScaleAni ? s.nNumFrame : 1;
				if(!WriteData32(pFile, ePODFileNodeAnimScale,	s.pNode[i].pfAnimScale,		iTransformationNo * 7))    return false;

				iTransformationNo = s.pNode[i].nAnimFlags & ePODHasMatrixAni ? s.nNumFrame : 1;
				if(!WriteData32(pFile, ePODFileNodeAnimMatrix,s.pNode[i].pfAnimMatrix,	iTransformationNo * 16))   return false;
			}

			if(!WriteMarker(pFile, ePODFileNode, true)) return false;
		}

		// Save: texture
		for(i = 0; i < s.nNumTexture; ++i)
		{
			if(!WriteMarker(pFile, ePODFileTexture, false)) return false;
			if(!WriteData(pFile, ePODFileTexName, s.pTexture[i].pszName, (unsigned int)strlen(s.pTexture[i].pszName)+1)) return false;
			if(!WriteMarker(pFile, ePODFileTexture, true)) return false;
		}
	}
	if(!WriteMarker(pFile, ePODFileScene, true)) return false;

	return true;
}

/****************************************************************************
** Local code: File reading
****************************************************************************/
/*!***************************************************************************
 @Function			ReadCPODData
 @Modified			s The CPODData to read into
 @Input				src CSource object to read data from.
 @Input				nSpec
 @Input				bValidData
 @Return			true if successful
 @Description		Read a CPODData block in  from a pod file
*****************************************************************************/
static bool ReadCPODData(
	CPODData			&s,
	CSource				&src,
	const unsigned int	nSpec,
	const bool			bValidData)
{
	unsigned int nName, nLen, nBuff;

	while(src.ReadMarker(nName, nLen))
	{
		if(nName == (nSpec | PVRTMODELPOD_TAG_END))
			return true;

		switch(nName)
		{
		case ePODFileDataType:	if(!src.Read32(s.eType)) return false;					break;
		case ePODFileN:			if(!src.Read32(s.n)) return false;						break;
		case ePODFileStride:	if(!src.Read32(s.nStride)) return false;					break;
		case ePODFileData:
			if(bValidData)
			{
				switch(PVRTModelPODDataTypeSize(s.eType))
				{
					case 1: if(!src.ReadAfterAlloc(s.pData, nLen)) return false; break;
					case 2: if(!src.ReadAfterAlloc16(s.pData, nLen)) return false; break;
					case 4: if(!src.ReadAfterAlloc32(s.pData, nLen)) return false; break;
					default:
						{ _ASSERT(false);}
				}
			}
			else
			{
				if(src.Read32(nBuff))
				{
					s.pData = (unsigned char*)nBuff;
				}
				else
				{
					return false;
				}
			}
		 break;

		default:
			if(!src.Skip(nLen)) return false;
		}
	}
	return false;
}

/*!***************************************************************************
 @Function			ReadCamera
 @Modified			s The SPODCamera to read into
 @Input				src	CSource object to read data from.
 @Return			true if successful
 @Description		Read a camera block in from a pod file
*****************************************************************************/
static bool ReadCamera(
	SPODCamera	&s,
	CSource		&src)
{
	unsigned int nName, nLen;
	s.pfAnimFOV = 0;

	while(src.ReadMarker(nName, nLen))
	{
		switch(nName)
		{
		case ePODFileCamera | PVRTMODELPOD_TAG_END:			return true;

		case ePODFileCamIdxTgt:		if(!src.Read32(s.nIdxTarget)) return false;					break;
		case ePODFileCamFOV:		if(!src.Read32(s.fFOV)) return false;							break;
		case ePODFileCamFar:		if(!src.Read32(s.fFar)) return false;							break;
		case ePODFileCamNear:		if(!src.Read32(s.fNear)) return false;						break;
		case ePODFileCamAnimFOV:	if(!src.ReadAfterAlloc32(s.pfAnimFOV, nLen)) return false;	break;

		default:
			if(!src.Skip(nLen)) return false;
		}
	}
	return false;
}

/*!***************************************************************************
 @Function			ReadLight
 @Modified			s The SPODLight to read into
 @Input				src	CSource object to read data from.
 @Return			true if successful
 @Description		Read a light block in from a pod file
*****************************************************************************/
static bool ReadLight(
	SPODLight	&s,
	CSource		&src)
{
	unsigned int nName, nLen;

	while(src.ReadMarker(nName, nLen))
	{
		switch(nName)
		{
		case ePODFileLight | PVRTMODELPOD_TAG_END:			return true;

		case ePODFileLightIdxTgt:	if(!src.Read32(s.nIdxTarget)) return false;	break;
		case ePODFileLightColour:	if(!src.ReadArray32((unsigned int*) s.pfColour, 3)) return false;		break;
		case ePODFileLightType:		if(!src.Read32(s.eType)) return false;		break;
		case ePODFileLightConstantAttenuation: 		if(!src.Read32(s.fConstantAttenuation))	return false;	break;
		case ePODFileLightLinearAttenuation:		if(!src.Read32(s.fLinearAttenuation))		return false;	break;
		case ePODFileLightQuadraticAttenuation:		if(!src.Read32(s.fQuadraticAttenuation))	return false;	break;
		case ePODFileLightFalloffAngle:				if(!src.Read32(s.fFalloffAngle))			return false;	break;
		case ePODFileLightFalloffExponent:			if(!src.Read32(s.fFalloffExponent))		return false;	break;
		default:
			if(!src.Skip(nLen)) return false;
		}
	}
	return false;
}

/*!***************************************************************************
 @Function			ReadMaterial
 @Modified			s The SPODMaterial to read into
 @Input				src	CSource object to read data from.
 @Return			true if successful
 @Description		Read a material block in from a pod file
*****************************************************************************/
static bool ReadMaterial(
	SPODMaterial	&s,
	CSource			&src)
{
	unsigned int nName, nLen;

	// Set texture IDs to -1
	s.nIdxTexDiffuse = -1;
	s.nIdxTexAmbient = -1;
	s.nIdxTexSpecularColour = -1;
	s.nIdxTexSpecularLevel = -1;
	s.nIdxTexBump = -1;
	s.nIdxTexEmissive = -1;
	s.nIdxTexGlossiness = -1;
	s.nIdxTexOpacity = -1;
	s.nIdxTexReflection = -1;
	s.nIdxTexRefraction = -1;

	// Set defaults for blend modes
	s.eBlendSrcRGB = s.eBlendSrcA = ePODBlendFunc_ONE;
	s.eBlendDstRGB = s.eBlendDstA = ePODBlendFunc_ZERO;
	s.eBlendOpRGB  = s.eBlendOpA  = ePODBlendOp_ADD;

	memset(s.pfBlendColour, 0, sizeof(s.pfBlendColour));
	memset(s.pfBlendFactor, 0, sizeof(s.pfBlendFactor));

	// Set default for material flags
	s.nFlags = 0;

	while(src.ReadMarker(nName, nLen))
	{
		switch(nName)
		{
		case ePODFileMaterial | PVRTMODELPOD_TAG_END:			return true;

		case ePODFileMatFlags:					if(!src.Read32(s.nFlags)) return false;				break;
		case ePODFileMatName:					if(!src.ReadAfterAlloc(s.pszName, nLen)) return false;		break;
		case ePODFileMatIdxTexDiffuse:			if(!src.Read32(s.nIdxTexDiffuse)) return false;				break;
		case ePODFileMatIdxTexAmbient:			if(!src.Read32(s.nIdxTexAmbient)) return false;				break;
		case ePODFileMatIdxTexSpecularColour:	if(!src.Read32(s.nIdxTexSpecularColour)) return false;		break;
		case ePODFileMatIdxTexSpecularLevel:	if(!src.Read32(s.nIdxTexSpecularLevel)) return false;			break;
		case ePODFileMatIdxTexBump:				if(!src.Read32(s.nIdxTexBump)) return false;					break;
		case ePODFileMatIdxTexEmissive:			if(!src.Read32(s.nIdxTexEmissive)) return false;				break;
		case ePODFileMatIdxTexGlossiness:		if(!src.Read32(s.nIdxTexGlossiness)) return false;			break;
		case ePODFileMatIdxTexOpacity:			if(!src.Read32(s.nIdxTexOpacity)) return false;				break;
		case ePODFileMatIdxTexReflection:		if(!src.Read32(s.nIdxTexReflection)) return false;			break;
		case ePODFileMatIdxTexRefraction:		if(!src.Read32(s.nIdxTexRefraction)) return false;			break;
		case ePODFileMatOpacity:		if(!src.Read32(s.fMatOpacity)) return false;						break;
		case ePODFileMatAmbient:		if(!src.ReadArray32((unsigned int*) s.pfMatAmbient,  sizeof(s.pfMatAmbient) / sizeof(*s.pfMatAmbient))) return false;		break;
		case ePODFileMatDiffuse:		if(!src.ReadArray32((unsigned int*) s.pfMatDiffuse,  sizeof(s.pfMatDiffuse) / sizeof(*s.pfMatDiffuse))) return false;		break;
		case ePODFileMatSpecular:		if(!src.ReadArray32((unsigned int*) s.pfMatSpecular, sizeof(s.pfMatSpecular) / sizeof(*s.pfMatSpecular))) return false;		break;
		case ePODFileMatShininess:		if(!src.Read32(s.fMatShininess)) return false;					break;
		case ePODFileMatEffectFile:		if(!src.ReadAfterAlloc(s.pszEffectFile, nLen)) return false;	break;
		case ePODFileMatEffectName:		if(!src.ReadAfterAlloc(s.pszEffectName, nLen)) return false;	break;
		case ePODFileMatBlendSrcRGB:	if(!src.Read32(s.eBlendSrcRGB))	return false;	break;
		case ePODFileMatBlendSrcA:		if(!src.Read32(s.eBlendSrcA))		return false;	break;
		case ePODFileMatBlendDstRGB:	if(!src.Read32(s.eBlendDstRGB))	return false;	break;
		case ePODFileMatBlendDstA:		if(!src.Read32(s.eBlendDstA))		return false;	break;
		case ePODFileMatBlendOpRGB:		if(!src.Read32(s.eBlendOpRGB))	return false;	break;
		case ePODFileMatBlendOpA:		if(!src.Read32(s.eBlendOpA))		return false;	break;
		case ePODFileMatBlendColour:	if(!src.ReadArray32((unsigned int*) s.pfBlendColour, sizeof(s.pfBlendColour) / sizeof(*s.pfBlendColour)))	return false;	break;
		case ePODFileMatBlendFactor:	if(!src.ReadArray32((unsigned int*) s.pfBlendFactor, sizeof(s.pfBlendFactor) / sizeof(*s.pfBlendFactor)))	return false;	break;

		default:
			if(!src.Skip(nLen)) return false;
		}
	}
	return false;
}

/*!***************************************************************************
 @Function			PVRTFixInterleavedEndiannessUsingCPODData
 @Modified			pInterleaved - The interleaved data
 @Input				data - The CPODData.
 @Return			ui32Size - Number of elements in pInterleaved
 @Description		Called multiple times and goes through the interleaved data
					correcting the endianness.
*****************************************************************************/
void PVRTFixInterleavedEndiannessUsingCPODData(unsigned char* pInterleaved, CPODData &data, unsigned int ui32Size)
{
	if(!data.n)
		return;

	size_t ui32TypeSize = PVRTModelPODDataTypeSize(data.eType);

	unsigned char ub[4];
	unsigned char *pData = pInterleaved + (size_t) data.pData;

	switch(ui32TypeSize)
	{
		case 1: return;
		case 2:
			{
				for(unsigned int i = 0; i < ui32Size; ++i)
				{
					for(unsigned int j = 0; j < data.n; ++j)
					{
						ub[0] = pData[ui32TypeSize * j + 0];
						ub[1] = pData[ui32TypeSize * j + 1];

						((unsigned short*) pData)[j] = (unsigned short) ((ub[1] << 8) | ub[0]);
					}

					pData += data.nStride;
				}
			}
			break;
		case 4:
			{
				for(unsigned int i = 0; i < ui32Size; ++i)
				{
					for(unsigned int j = 0; j < data.n; ++j)
					{
						ub[0] = pData[ui32TypeSize * j + 0];
						ub[1] = pData[ui32TypeSize * j + 1];
						ub[2] = pData[ui32TypeSize * j + 2];
						ub[3] = pData[ui32TypeSize * j + 3];

						((unsigned int*) pData)[j] = (unsigned int) ((ub[3] << 24) | (ub[2] << 16) | (ub[1] << 8) | ub[0]);
					}

					pData += data.nStride;
				}
			}
			break;
		default: { _ASSERT(false); }
	};
}

void PVRTFixInterleavedEndianness(SPODMesh &s)
{
	if(!s.pInterleaved || PVRTIsLittleEndian())
		return;

	PVRTFixInterleavedEndiannessUsingCPODData(s.pInterleaved, s.sVertex, s.nNumVertex);
	PVRTFixInterleavedEndiannessUsingCPODData(s.pInterleaved, s.sNormals, s.nNumVertex);
	PVRTFixInterleavedEndiannessUsingCPODData(s.pInterleaved, s.sTangents, s.nNumVertex);
	PVRTFixInterleavedEndiannessUsingCPODData(s.pInterleaved, s.sBinormals, s.nNumVertex);

	for(unsigned int i = 0; i < s.nNumUVW; ++i)
		PVRTFixInterleavedEndiannessUsingCPODData(s.pInterleaved, s.psUVW[i], s.nNumVertex);

	PVRTFixInterleavedEndiannessUsingCPODData(s.pInterleaved, s.sVtxColours, s.nNumVertex);
	PVRTFixInterleavedEndiannessUsingCPODData(s.pInterleaved, s.sBoneIdx, s.nNumVertex);
	PVRTFixInterleavedEndiannessUsingCPODData(s.pInterleaved, s.sBoneWeight, s.nNumVertex);
}

/*!***************************************************************************
 @Function			ReadMesh
 @Modified			s The SPODMesh to read into
 @Input				src	CSource object to read data from.
 @Return			true if successful
 @Description		Read a mesh block in from a pod file
*****************************************************************************/
static bool ReadMesh(
	SPODMesh	&s,
	CSource		&src)
{
	unsigned int	nName, nLen;
	unsigned int	nUVWs=0;

	while(src.ReadMarker(nName, nLen))
	{
		switch(nName)
		{
		case ePODFileMesh | PVRTMODELPOD_TAG_END:
			if(nUVWs != s.nNumUVW)
				return false;
			PVRTFixInterleavedEndianness(s);
			return true;

		case ePODFileMeshNumVtx:			if(!src.Read32(s.nNumVertex)) return false;													break;
		case ePODFileMeshNumFaces:			if(!src.Read32(s.nNumFaces)) return false;													break;
		case ePODFileMeshNumUVW:			if(!src.Read32(s.nNumUVW)) return false;	if(!SafeAlloc(s.psUVW, s.nNumUVW)) return false;	break;
		case ePODFileMeshStripLength:		if(!src.ReadAfterAlloc32(s.pnStripLength, nLen)) return false;								break;
		case ePODFileMeshNumStrips:			if(!src.Read32(s.nNumStrips)) return false;													break;
		case ePODFileMeshInterleaved:		if(!src.ReadAfterAlloc(s.pInterleaved, nLen)) return false;									break;
		case ePODFileMeshBoneBatches:		if(!src.ReadAfterAlloc32(s.sBoneBatches.pnBatches, nLen)) return false;						break;
		case ePODFileMeshBoneBatchBoneCnts:	if(!src.ReadAfterAlloc32(s.sBoneBatches.pnBatchBoneCnt, nLen)) return false;					break;
		case ePODFileMeshBoneBatchOffsets:	if(!src.ReadAfterAlloc32(s.sBoneBatches.pnBatchOffset, nLen)) return false;					break;
		case ePODFileMeshBoneBatchBoneMax:	if(!src.Read32(s.sBoneBatches.nBatchBoneMax)) return false;									break;
		case ePODFileMeshBoneBatchCnt:		if(!src.Read32(s.sBoneBatches.nBatchCnt)) return false;										break;

		case ePODFileMeshFaces:			if(!ReadCPODData(s.sFaces, src, ePODFileMeshFaces, true)) return false;			break;
		case ePODFileMeshVtx:			if(!ReadCPODData(s.sVertex, src, ePODFileMeshVtx, s.pInterleaved == 0)) return false;			break;
		case ePODFileMeshNor:			if(!ReadCPODData(s.sNormals, src, ePODFileMeshNor, s.pInterleaved == 0)) return false;			break;
		case ePODFileMeshTan:			if(!ReadCPODData(s.sTangents, src, ePODFileMeshTan, s.pInterleaved == 0)) return false;			break;
		case ePODFileMeshBin:			if(!ReadCPODData(s.sBinormals, src, ePODFileMeshBin, s.pInterleaved == 0)) return false;			break;
		case ePODFileMeshUVW:			if(!ReadCPODData(s.psUVW[nUVWs++], src, ePODFileMeshUVW, s.pInterleaved == 0)) return false;		break;
		case ePODFileMeshVtxCol:		if(!ReadCPODData(s.sVtxColours, src, ePODFileMeshVtxCol, s.pInterleaved == 0)) return false;		break;
		case ePODFileMeshBoneIdx:		if(!ReadCPODData(s.sBoneIdx, src, ePODFileMeshBoneIdx, s.pInterleaved == 0)) return false;		break;
		case ePODFileMeshBoneWeight:	if(!ReadCPODData(s.sBoneWeight, src, ePODFileMeshBoneWeight, s.pInterleaved == 0)) return false;	break;

		default:
			if(!src.Skip(nLen)) return false;
		}
	}
	return false;
}

/*!***************************************************************************
 @Function			ReadNode
 @Modified			s The SPODNode to read into
 @Input				src	CSource object to read data from.
 @Return			true if successful
 @Description		Read a node block in from a pod file
*****************************************************************************/
static bool ReadNode(
	SPODNode	&s,
	CSource		&src)
{
	unsigned int nName, nLen;
	bool bOldNodeFormat = false;
	VERTTYPE fPos[3]   = {0,0,0};
	VERTTYPE fQuat[4]  = {0,0,0,f2vt(1)};
	VERTTYPE fScale[7] = {f2vt(1),f2vt(1),f2vt(1),0,0,0,0};

	while(src.ReadMarker(nName, nLen))
	{
		switch(nName)
		{
		case ePODFileNode | PVRTMODELPOD_TAG_END:
			if(bOldNodeFormat)
			{
				if(s.pfAnimPosition)
					s.nAnimFlags |= ePODHasPositionAni;
				else
				{
					s.pfAnimPosition = (VERTTYPE*) malloc(sizeof(fPos));
					memcpy(s.pfAnimPosition, fPos, sizeof(fPos));
				}

				if(s.pfAnimRotation)
					s.nAnimFlags |= ePODHasRotationAni;
				else
				{
					s.pfAnimRotation = (VERTTYPE*) malloc(sizeof(fQuat));
					memcpy(s.pfAnimRotation, fQuat, sizeof(fQuat));
				}

				if(s.pfAnimScale)
					s.nAnimFlags |= ePODHasScaleAni;
				else
				{
					s.pfAnimScale = (VERTTYPE*) malloc(sizeof(fScale));
					memcpy(s.pfAnimScale, fScale, sizeof(fScale));
				}
			}
			return true;

		case ePODFileNodeIdx:		if(!src.Read32(s.nIdx)) return false;								break;
		case ePODFileNodeName:		if(!src.ReadAfterAlloc(s.pszName, nLen)) return false;			break;
		case ePODFileNodeIdxMat:	if(!src.Read32(s.nIdxMaterial)) return false;						break;
		case ePODFileNodeIdxParent:	if(!src.Read32(s.nIdxParent)) return false;						break;
		case ePODFileNodeAnimFlags:if(!src.Read32(s.nAnimFlags))return false;							break;
		case ePODFileNodeAnimPos:	if(!src.ReadAfterAlloc32(s.pfAnimPosition, nLen)) return false;	break;
		case ePODFileNodeAnimRot:	if(!src.ReadAfterAlloc32(s.pfAnimRotation, nLen)) return false;	break;
		case ePODFileNodeAnimScale:	if(!src.ReadAfterAlloc32(s.pfAnimScale, nLen)) return false;		break;
		case ePODFileNodeAnimMatrix:if(!src.ReadAfterAlloc32(s.pfAnimMatrix, nLen)) return false;	break;

		// Parameters from the older pod format
		case ePODFileNodePos:		if(!src.ReadArray32((unsigned int*) fPos, 3))   return false;		bOldNodeFormat = true;		break;
		case ePODFileNodeRot:		if(!src.ReadArray32((unsigned int*) fQuat, 4))  return false;		bOldNodeFormat = true;		break;
		case ePODFileNodeScale:		if(!src.ReadArray32((unsigned int*) fScale,3)) return false;		bOldNodeFormat = true;		break;

		default:
			if(!src.Skip(nLen)) return false;
		}
	}

	return false;
}

/*!***************************************************************************
 @Function			ReadTexture
 @Modified			s The SPODTexture to read into
 @Input				src	CSource object to read data from.
 @Return			true if successful
 @Description		Read a texture block in from a pod file
*****************************************************************************/
static bool ReadTexture(
	SPODTexture	&s,
	CSource		&src)
{
	unsigned int nName, nLen;

	while(src.ReadMarker(nName, nLen))
	{
		switch(nName)
		{
		case ePODFileTexture | PVRTMODELPOD_TAG_END:			return true;

		case ePODFileTexName:		if(!src.ReadAfterAlloc(s.pszName, nLen)) return false;			break;

		default:
			if(!src.Skip(nLen)) return false;
		}
	}
	return false;
}

/*!***************************************************************************
 @Function			ReadScene
 @Modified			s The SPODScene to read into
 @Input				src	CSource object to read data from.
 @Return			true if successful
 @Description		Read a scene block in from a pod file
*****************************************************************************/
static bool ReadScene(
	SPODScene	&s,
	CSource		&src)
{
	unsigned int nName, nLen;
	unsigned int nCameras=0, nLights=0, nMaterials=0, nMeshes=0, nTextures=0, nNodes=0;

	while(src.ReadMarker(nName, nLen))
	{
		switch(nName)
		{
		case ePODFileScene | PVRTMODELPOD_TAG_END:
			if(nCameras		!= s.nNumCamera) return false;
			if(nLights		!= s.nNumLight) return false;
			if(nMaterials	!= s.nNumMaterial) return false;
			if(nMeshes		!= s.nNumMesh) return false;
			if(nTextures	!= s.nNumTexture) return false;
			if(nNodes		!= s.nNumNode) return false;
			return true;

		case ePODFileColourBackground:	if(!src.ReadArray32((unsigned int*) s.pfColourBackground, sizeof(s.pfColourBackground) / sizeof(*s.pfColourBackground))) return false;	break;
		case ePODFileColourAmbient:		if(!src.ReadArray32((unsigned int*) s.pfColourAmbient, sizeof(s.pfColourAmbient) / sizeof(*s.pfColourAmbient))) return false;		break;
		case ePODFileNumCamera:			if(!src.Read32(s.nNumCamera)) return false;			if(!SafeAlloc(s.pCamera, s.nNumCamera)) return false;		break;
		case ePODFileNumLight:			if(!src.Read32(s.nNumLight)) return false;			if(!SafeAlloc(s.pLight, s.nNumLight)) return false;			break;
		case ePODFileNumMesh:			if(!src.Read32(s.nNumMesh)) return false;				if(!SafeAlloc(s.pMesh, s.nNumMesh)) return false;			break;
		case ePODFileNumNode:			if(!src.Read32(s.nNumNode)) return false;				if(!SafeAlloc(s.pNode, s.nNumNode)) return false;			break;
		case ePODFileNumMeshNode:		if(!src.Read32(s.nNumMeshNode)) return false;			break;
		case ePODFileNumTexture:		if(!src.Read32(s.nNumTexture)) return false;			if(!SafeAlloc(s.pTexture, s.nNumTexture)) return false;		break;
		case ePODFileNumMaterial:		if(!src.Read32(s.nNumMaterial)) return false;			if(!SafeAlloc(s.pMaterial, s.nNumMaterial)) return false;	break;
		case ePODFileNumFrame:			if(!src.Read32(s.nNumFrame)) return false;			break;
		case ePODFileFlags:				if(!src.Read32(s.nFlags)) return false;				break;

		case ePODFileCamera:	if(!ReadCamera(s.pCamera[nCameras++], src)) return false;		break;
		case ePODFileLight:		if(!ReadLight(s.pLight[nLights++], src)) return false;			break;
		case ePODFileMaterial:	if(!ReadMaterial(s.pMaterial[nMaterials++], src)) return false;	break;
		case ePODFileMesh:		if(!ReadMesh(s.pMesh[nMeshes++], src)) return false;			break;
		case ePODFileNode:		if(!ReadNode(s.pNode[nNodes++], src)) return false;				break;
		case ePODFileTexture:	if(!ReadTexture(s.pTexture[nTextures++], src)) return false;	break;

		default:
			if(!src.Skip(nLen)) return false;
		}
	}
	return false;
}

/*!***************************************************************************
 @Function			Read
 @Output			pS				SPODScene data. May be NULL.
 @Input				src				CSource object to read data from.
 @Output			pszExpOpt		Export options.
 @Input				count			Data size.
 @Output			pszHistory		Export history.
 @Input				historyCount	History data size.
 @Description		Loads the specified ".POD" file; returns the scene in
					pScene. This structure must later be destroyed with
					PVRTModelPODDestroy() to prevent memory leaks.
					".POD" files are exported from 3D Studio MAX using a
					PowerVR plugin. pS may be NULL if only the export options
					are required.
*****************************************************************************/
static bool Read(
	SPODScene		* const pS,
	CSource			&src,
	char			* const pszExpOpt,
	const size_t	count,
	char			* const pszHistory,
	const size_t	historyCount)
{
	unsigned int	nName, nLen;
	bool			bVersionOK = false, bDone = false;
	bool			bNeedOptions = pszExpOpt != 0;
	bool			bNeedHistory = pszHistory != 0;
	bool			bLoadingOptionsOrHistory = bNeedOptions || bNeedHistory;

	while(src.ReadMarker(nName, nLen))
	{
		switch(nName)
		{
		case ePODFileVersion:
			{
				char *pszVersion = NULL;
				if(nLen != strlen(PVRTMODELPOD_VERSION)+1) return false;
				if(!SafeAlloc(pszVersion, nLen)) return false;
				if(!src.Read(pszVersion, nLen)) return false;
				if(strcmp(pszVersion, PVRTMODELPOD_VERSION) != 0) return false;
				bVersionOK = true;
				FREE(pszVersion);
			}
			continue;

		case ePODFileScene:
			if(pS)
			{
				if(!ReadScene(*pS, src))
					return false;
				bDone = true;
			}
			continue;

		case ePODFileExpOpt:
			if(bNeedOptions)
			{
				if(!src.Read(pszExpOpt, PVRT_MIN(nLen, (unsigned int) count)))
					return false;

				bNeedOptions = false;

				if(count < nLen)
					nLen -= (unsigned int) count ; // Adjust nLen as the read has moved our position
				else
					nLen = 0;
			}
			break;

		case ePODFileHistory:
			if(bNeedHistory)
			{
				if(!src.Read(pszHistory, PVRT_MIN(nLen, (unsigned int) historyCount)))
					return false;

				bNeedHistory = false;

				if(count < nLen)
					nLen -= (unsigned int) historyCount; // Adjust nLen as the read has moved our position
				else
					nLen = 0;
			}
			break;

		case ePODFileScene | PVRTMODELPOD_TAG_END:
			return bVersionOK == true && bDone == true;

		case (unsigned int) ePODFileEndiannessMisMatch:
			PVRTErrorOutputDebug("Error: Endianness mismatch between the .pod file and the platform.\n");
			return false;

		}

		if(bLoadingOptionsOrHistory && !bNeedOptions && !bNeedHistory)
			return true; // The options and/or history has been loaded

		// Unhandled data, skip it
		if(!src.Skip(nLen))
			return false;
	}

	if(bLoadingOptionsOrHistory)
		return true;

	/*
		Convert data to fixed or float point as this build desires
	*/
#ifdef PVRT_FIXED_POINT_ENABLE
	if(!(pS->nFlags & PVRTMODELPODSF_FIXED))
		PVRTModelPODToggleFixedPoint(*pS);
#else
	if(pS->nFlags & PVRTMODELPODSF_FIXED)
		PVRTModelPODToggleFixedPoint(*pS);
#endif

	return bVersionOK == true && bDone == true;
}

/*!***************************************************************************
 @Function			ReadFromSourceStream
 @Output			pS				CPVRTModelPOD data. May not be NULL.
 @Input				src				CSource object to read data from.
 @Output			pszExpOpt		Export options.
 @Input				count			Data size.
 @Output			pszHistory		Export history.
 @Input				historyCount	History data size.
 @Description		Loads the ".POD" data from the source stream; returns the scene
					in pS.
*****************************************************************************/
static EPVRTError ReadFromSourceStream(
	CPVRTModelPOD	* const pS,
	CSourceStream &src,
	char			* const pszExpOpt,
	const size_t	count,
	char			* const pszHistory,
	const size_t	historyCount)
{
	memset(pS, 0, sizeof(*pS));
	if(!Read(pszExpOpt || pszHistory ? NULL : pS, src, pszExpOpt, count, pszHistory, historyCount))
		return PVR_FAIL;

	if(pS->InitImpl() != PVR_SUCCESS)
		return PVR_FAIL;

	return PVR_SUCCESS;
}

/****************************************************************************
** Class: CPVRTModelPOD
****************************************************************************/

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
EPVRTError CPVRTModelPOD::ReadFromFile(
	const char		* const pszFileName,
	char			* const pszExpOpt,
	const size_t	count,
	char			* const pszHistory,
	const size_t	historyCount)
{
	CSourceStream src;

	if(!src.Init(pszFileName))
		return PVR_FAIL;

	return ReadFromSourceStream(this, src, pszExpOpt, count, pszHistory, historyCount);
}

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
EPVRTError CPVRTModelPOD::ReadFromMemory(
	const char		* pData,
	const size_t	i32Size,
	char			* const pszExpOpt,
	const size_t	count,
	char			* const pszHistory,
	const size_t	historyCount)
{
	CSourceStream src;

	if(!src.Init(pData, i32Size))
		return PVR_FAIL;

	return ReadFromSourceStream(this, src, pszExpOpt, count, pszHistory, historyCount);
}

/*!***************************************************************************
 @Function			ReadFromMemory
 @Input				scene			Scene data from the header file
 @Return			PVR_SUCCESS if successful, PVR_FAIL if not
 @Description		Sets the scene data from the supplied data structure. Use
					when loading from .H files.
*****************************************************************************/
EPVRTError CPVRTModelPOD::ReadFromMemory(
	const SPODScene &scene)
{
	Destroy();

	memset(this, 0, sizeof(*this));

	*(SPODScene*)this = scene;

	if(InitImpl() != PVR_SUCCESS)
		return PVR_FAIL;

	m_pImpl->bFromMemory = true;

	return PVR_SUCCESS;
}

/*!***************************************************************************
 @Function			CopyCPODData
 @Input				Target
 @Input				Source
 @Input				ui32No
 @Input				bInterleaved
 @Description		Used by CopyFromMemory to copy the CPODData
*****************************************************************************/
void CopyCPODData(CPODData &Target, CPODData &Source, unsigned int ui32No, bool bInterleaved)
{
	FREE(Target.pData);

	Target.eType	= Source.eType;
	Target.n		= Source.n;
	Target.nStride  = Source.nStride;

	if(bInterleaved)
	{
		Target.pData = Source.pData;
	}
	else if(Source.pData)
	{
		size_t ui32Size = PVRTModelPODDataStride(Target) * ui32No;

		if(SafeAlloc(Target.pData, ui32Size))
			memcpy(Target.pData, Source.pData, ui32Size);
	}
}

/*!***************************************************************************
 @Function			CopyFromMemory
 @Input				scene			Scene data
 @Return			PVR_SUCCESS if successful, PVR_FAIL if not
 @Description		Sets the scene data from the supplied data structure.
*****************************************************************************/
EPVRTError CPVRTModelPOD::CopyFromMemory(const SPODScene &scene)
{
	Destroy();

	unsigned int i,j;

	// SPODScene
	nNumFrame	= scene.nNumFrame;
	nFlags		= scene.nFlags;

	for(i = 0; i < 3; ++i)
	{
		pfColourBackground[i] = scene.pfColourBackground[i];
		pfColourAmbient[i]	  = scene.pfColourAmbient[i];
	}

	// Nodes
	if(scene.nNumNode && SafeAlloc(pNode, sizeof(SPODNode) * scene.nNumNode))
	{
		nNumNode     = scene.nNumNode;
		nNumMeshNode = scene.nNumMeshNode;

		for(i = 0; i < nNumNode; ++i)
		{
			pNode[i].nIdx = scene.pNode[i].nIdx;
			pNode[i].nIdxMaterial = scene.pNode[i].nIdxMaterial;
			pNode[i].nIdxParent = scene.pNode[i].nIdxParent;
			pNode[i].nAnimFlags = scene.pNode[i].nAnimFlags;

			if(scene.pNode[i].pszName && SafeAlloc(pNode[i].pszName, strlen(scene.pNode[i].pszName) + 1))
				memcpy(pNode[i].pszName, scene.pNode[i].pszName, strlen(scene.pNode[i].pszName) + 1);

			int i32Size;

			i32Size = scene.pNode[i].nAnimFlags & ePODHasPositionAni ? scene.nNumFrame : 1;

			if(scene.pNode[i].pfAnimPosition && SafeAlloc(pNode[i].pfAnimPosition, sizeof(*pNode[i].pfAnimPosition) * i32Size * 3))
				memcpy(pNode[i].pfAnimPosition, scene.pNode[i].pfAnimPosition, sizeof(*pNode[i].pfAnimPosition) * i32Size * 3);

			i32Size = scene.pNode[i].nAnimFlags & ePODHasRotationAni ? scene.nNumFrame : 1;

			if(scene.pNode[i].pfAnimRotation && SafeAlloc(pNode[i].pfAnimRotation, sizeof(*pNode[i].pfAnimRotation) * i32Size * 4))
				memcpy(pNode[i].pfAnimRotation, scene.pNode[i].pfAnimRotation, sizeof(*pNode[i].pfAnimRotation) * i32Size * 4);

			i32Size = scene.pNode[i].nAnimFlags & ePODHasScaleAni ? scene.nNumFrame : 1;

			if(scene.pNode[i].pfAnimScale && SafeAlloc(pNode[i].pfAnimScale, sizeof(*pNode[i].pfAnimScale) * i32Size * 7))
				memcpy(pNode[i].pfAnimScale, scene.pNode[i].pfAnimScale, sizeof(*pNode[i].pfAnimScale) * i32Size * 7);

			i32Size = scene.pNode[i].nAnimFlags & ePODHasMatrixAni ? scene.nNumFrame : 1;

			if(scene.pNode[i].pfAnimMatrix && SafeAlloc(pNode[i].pfAnimMatrix, sizeof(*pNode[i].pfAnimMatrix) * i32Size * 16))
				memcpy(pNode[i].pfAnimMatrix, scene.pNode[i].pfAnimMatrix, sizeof(*pNode[i].pfAnimMatrix) * i32Size * 16);
		}
	}

	// Meshes
	if(scene.nNumMesh && SafeAlloc(pMesh, sizeof(SPODMesh) * scene.nNumMesh))
	{
		nNumMesh = scene.nNumMesh;

		for(i = 0; i < nNumMesh; ++i)
		{
			size_t  i32Stride = 0;
			bool bInterleaved = scene.pMesh[i].pInterleaved != 0;
			pMesh[i].nNumVertex = scene.pMesh[i].nNumVertex;
			pMesh[i].nNumFaces  = scene.pMesh[i].nNumFaces;

			// Face data
			CopyCPODData(pMesh[i].sFaces	 , scene.pMesh[i].sFaces	 , pMesh[i].nNumFaces * 3, false);

			// Vertex data
			CopyCPODData(pMesh[i].sVertex	 , scene.pMesh[i].sVertex	 , pMesh[i].nNumVertex, bInterleaved);
			i32Stride += PVRTModelPODDataStride(pMesh[i].sVertex);

			CopyCPODData(pMesh[i].sNormals	 , scene.pMesh[i].sNormals	 , pMesh[i].nNumVertex, bInterleaved);
			i32Stride += PVRTModelPODDataStride(pMesh[i].sNormals);

			CopyCPODData(pMesh[i].sTangents	 , scene.pMesh[i].sTangents	 , pMesh[i].nNumVertex, bInterleaved);
			i32Stride += PVRTModelPODDataStride(pMesh[i].sTangents);

			CopyCPODData(pMesh[i].sBinormals , scene.pMesh[i].sBinormals , pMesh[i].nNumVertex, bInterleaved);
			i32Stride += PVRTModelPODDataStride(pMesh[i].sBinormals);

			CopyCPODData(pMesh[i].sVtxColours, scene.pMesh[i].sVtxColours, pMesh[i].nNumVertex, bInterleaved);
			i32Stride += PVRTModelPODDataStride(pMesh[i].sVtxColours);

			CopyCPODData(pMesh[i].sBoneIdx	 , scene.pMesh[i].sBoneIdx	 , pMesh[i].nNumVertex, bInterleaved);
			i32Stride += PVRTModelPODDataStride(pMesh[i].sBoneIdx);

			CopyCPODData(pMesh[i].sBoneWeight, scene.pMesh[i].sBoneWeight, pMesh[i].nNumVertex, bInterleaved);
			i32Stride += PVRTModelPODDataStride(pMesh[i].sBoneWeight);

			if(scene.pMesh[i].nNumUVW && SafeAlloc(pMesh[i].psUVW, sizeof(CPODData) * scene.pMesh[i].nNumUVW))
			{
				pMesh[i].nNumUVW = scene.pMesh[i].nNumUVW;

				for(j = 0; j < pMesh[i].nNumUVW; ++j)
				{
					CopyCPODData(pMesh[i].psUVW[j], scene.pMesh[i].psUVW[j], pMesh[i].nNumVertex, bInterleaved);
					i32Stride += PVRTModelPODDataStride(pMesh[i].psUVW[j]);
				}
			}

			// Allocate and copy interleaved array
			if(bInterleaved && SafeAlloc(pMesh[i].pInterleaved, pMesh[i].nNumVertex * i32Stride))
				memcpy(pMesh[i].pInterleaved, scene.pMesh[i].pInterleaved, pMesh[i].nNumVertex * i32Stride);

			if(scene.pMesh[i].pnStripLength && SafeAlloc(pMesh[i].pnStripLength, sizeof(*pMesh[i].pnStripLength) * pMesh[i].nNumFaces))
			{
				memcpy(pMesh[i].pnStripLength, scene.pMesh[i].pnStripLength, sizeof(*pMesh[i].pnStripLength) * pMesh[i].nNumFaces);
				pMesh[i].nNumStrips = scene.pMesh[i].nNumStrips;
			}

			if(scene.pMesh[i].sBoneBatches.nBatchCnt)
			{
				pMesh[i].sBoneBatches.Release();

				pMesh[i].sBoneBatches.nBatchBoneMax = scene.pMesh[i].sBoneBatches.nBatchBoneMax;
				pMesh[i].sBoneBatches.nBatchCnt     = scene.pMesh[i].sBoneBatches.nBatchCnt;

				if(scene.pMesh[i].sBoneBatches.pnBatches)
				{
					pMesh[i].sBoneBatches.pnBatches = new int[pMesh[i].sBoneBatches.nBatchCnt * pMesh[i].sBoneBatches.nBatchBoneMax];

					if(pMesh[i].sBoneBatches.pnBatches)
						memcpy(pMesh[i].sBoneBatches.pnBatches, scene.pMesh[i].sBoneBatches.pnBatches, pMesh[i].sBoneBatches.nBatchCnt * pMesh[i].sBoneBatches.nBatchBoneMax * sizeof(*pMesh[i].sBoneBatches.pnBatches));
				}

				if(scene.pMesh[i].sBoneBatches.pnBatchBoneCnt)
				{
					pMesh[i].sBoneBatches.pnBatchBoneCnt = new int[pMesh[i].sBoneBatches.nBatchCnt];

					if(pMesh[i].sBoneBatches.pnBatchBoneCnt)
						memcpy(pMesh[i].sBoneBatches.pnBatchBoneCnt, scene.pMesh[i].sBoneBatches.pnBatchBoneCnt, pMesh[i].sBoneBatches.nBatchCnt * sizeof(*pMesh[i].sBoneBatches.pnBatchBoneCnt));
				}

				if(scene.pMesh[i].sBoneBatches.pnBatchOffset)
				{
					pMesh[i].sBoneBatches.pnBatchOffset = new int[pMesh[i].sBoneBatches.nBatchCnt];

					if(pMesh[i].sBoneBatches.pnBatchOffset)
						memcpy(pMesh[i].sBoneBatches.pnBatchOffset, scene.pMesh[i].sBoneBatches.pnBatchOffset, pMesh[i].sBoneBatches.nBatchCnt * sizeof(*pMesh[i].sBoneBatches.pnBatchOffset));
				}
			}

			pMesh[i].ePrimitiveType = scene.pMesh[i].ePrimitiveType;
		}
	}

	// Cameras
	if(scene.nNumCamera && SafeAlloc(pCamera, sizeof(SPODCamera) * scene.nNumCamera))
	{
		nNumCamera = scene.nNumCamera;

		for(i = 0; i < nNumCamera; ++i)
		{
			pCamera[i].nIdxTarget = scene.pCamera[i].nIdxTarget;
			pCamera[i].fNear = scene.pCamera[i].fNear;
			pCamera[i].fFar  = scene.pCamera[i].fFar;
			pCamera[i].fFOV  = scene.pCamera[i].fFOV;

			if(scene.pCamera[i].pfAnimFOV && SafeAlloc(pCamera[i].pfAnimFOV, sizeof(*pCamera[i].pfAnimFOV) * scene.nNumFrame))
				memcpy(pCamera[i].pfAnimFOV, scene.pCamera[i].pfAnimFOV, sizeof(*pCamera[i].pfAnimFOV) * scene.nNumFrame);
		}
	}

	// Lights
	if(scene.nNumLight && SafeAlloc(pLight, sizeof(SPODLight) * scene.nNumLight))
	{
		nNumLight = scene.nNumLight;

		for(i = 0; i < nNumLight; ++i)
		{
			memcpy(&pLight[i], &scene.pLight[i], sizeof(SPODLight));
		}
	}

	// Textures
	if(scene.nNumTexture && SafeAlloc(pTexture, sizeof(SPODTexture) * scene.nNumTexture))
	{
		nNumTexture = scene.nNumTexture;

		for(i = 0; i < nNumTexture; ++i)
		{
			if(scene.pTexture[i].pszName && SafeAlloc(pTexture[i].pszName, strlen(scene.pTexture[i].pszName) + 1))
				memcpy(pTexture[i].pszName, scene.pTexture[i].pszName, strlen(scene.pTexture[i].pszName) + 1);
		}
	}

	// Materials
	if(scene.nNumMaterial && SafeAlloc(pMaterial, sizeof(SPODMaterial) * scene.nNumMaterial))
	{
		nNumMaterial = scene.nNumMaterial;

		for(i = 0; i < nNumMaterial; ++i)
		{
			memcpy(&pMaterial[i], &scene.pMaterial[i], sizeof(SPODMaterial));

			pMaterial[i].pszName = 0;
			pMaterial[i].pszEffectFile = 0;
			pMaterial[i].pszEffectName = 0;

			if(scene.pMaterial[i].pszName && SafeAlloc(pMaterial[i].pszName, strlen(scene.pMaterial[i].pszName) + 1))
				memcpy(pMaterial[i].pszName, scene.pMaterial[i].pszName, strlen(scene.pMaterial[i].pszName) + 1);

			if(scene.pMaterial[i].pszEffectFile && SafeAlloc(pMaterial[i].pszEffectFile, strlen(scene.pMaterial[i].pszEffectFile) + 1))
				memcpy(pMaterial[i].pszEffectFile, scene.pMaterial[i].pszEffectFile, strlen(scene.pMaterial[i].pszEffectFile) + 1);

			if(scene.pMaterial[i].pszEffectName && SafeAlloc(pMaterial[i].pszEffectName, strlen(scene.pMaterial[i].pszEffectName) + 1))
				memcpy(pMaterial[i].pszEffectName, scene.pMaterial[i].pszEffectName, strlen(scene.pMaterial[i].pszEffectName) + 1);
		}
	}

	if(InitImpl() != PVR_SUCCESS)
		return PVR_FAIL;

	return PVR_SUCCESS;
}

#ifdef WIN32
/*!***************************************************************************
 @Function			ReadFromResource
 @Input				pszName			Name of the resource to load from
 @Return			PVR_SUCCESS if successful, PVR_FAIL if not
 @Description		Loads the specified ".POD" file; returns the scene in
					pScene. This structure must later be destroyed with
					PVRTModelPODDestroy() to prevent memory leaks.
					".POD" files are exported from 3D Studio MAX using a
					PowerVR plugin.
*****************************************************************************/
EPVRTError CPVRTModelPOD::ReadFromResource(
	const TCHAR * const pszName)
{
	CSourceResource src;

	if(!src.Init(pszName))
		return PVR_FAIL;

	memset(this, 0, sizeof(*this));
	if(!Read(this, src, NULL, 0, NULL, 0))
		return PVR_FAIL;
	if(InitImpl() != PVR_SUCCESS)
		return PVR_FAIL;
	return PVR_SUCCESS;
}
#endif /* WIN32 */

/*!***********************************************************************
 @Function		InitImpl
 @Description	Used by the Read*() fns to initialise implementation
				details. Should also be called by applications which
				manually build data in the POD structures for rendering;
				in this case call it after the data has been created.
				Otherwise, do not call this function.
*************************************************************************/
EPVRTError CPVRTModelPOD::InitImpl()
{
	// Allocate space for implementation data
	m_pImpl = new SPVRTPODImpl;
	if(!m_pImpl)
		return PVR_FAIL;

	// Zero implementation data
	memset(m_pImpl, 0, sizeof(*m_pImpl));

#ifdef _DEBUG
	m_pImpl->nWmTotal = 0;
#endif

	// Allocate world-matrix cache
	m_pImpl->pfCache		= new VERTTYPE[nNumNode];
	m_pImpl->pWmCache		= new PVRTMATRIX[nNumNode];
	m_pImpl->pWmZeroCache	= new PVRTMATRIX[nNumNode];
	FlushCache();

	return PVR_SUCCESS;
}

/*!***********************************************************************
 @Function		DestroyImpl
 @Description	Used to free memory allocated by the implementation.
*************************************************************************/
void CPVRTModelPOD::DestroyImpl()
{
	if(m_pImpl)
	{
		if(m_pImpl->pfCache)		delete [] m_pImpl->pfCache;
		if(m_pImpl->pWmCache)		delete [] m_pImpl->pWmCache;
		if(m_pImpl->pWmZeroCache)	delete [] m_pImpl->pWmZeroCache;

		delete m_pImpl;
		m_pImpl = 0;
	}
}

/*!***********************************************************************
 @Function		FlushCache
 @Description	Clears the matrix cache; use this if necessary when you
				edit the position or animation of a node.
*************************************************************************/
void CPVRTModelPOD::FlushCache()
{
	// Pre-calc frame zero matrices
	SetFrame(0);
	for(unsigned int i = 0; i < nNumNode; ++i)
		GetWorldMatrixNoCache(m_pImpl->pWmZeroCache[i], pNode[i]);

	// Load cache with frame-zero data
	memcpy(m_pImpl->pWmCache, m_pImpl->pWmZeroCache, nNumNode * sizeof(*m_pImpl->pWmCache));
	memset(m_pImpl->pfCache, 0, nNumNode * sizeof(*m_pImpl->pfCache));
}

/*!***************************************************************************
 @Function			Constructor
 @Description		Initializes the pointer to scene data to NULL
*****************************************************************************/
CPVRTModelPOD::CPVRTModelPOD() : m_pImpl(NULL)
{}

/*!***************************************************************************
 @Function			Destructor
 @Description		Frees the memory allocated to store the scene in pScene.
*****************************************************************************/
CPVRTModelPOD::~CPVRTModelPOD()
{
	Destroy();
}

/*!***************************************************************************
 @Function			Destroy
 @Description		Frees the memory allocated to store the scene in pScene.
*****************************************************************************/
void CPVRTModelPOD::Destroy()
{
	unsigned int	i;

	if(m_pImpl != NULL)
	{
		/*
			Only attempt to free this memory if it was actually allocated at
			run-time, as opposed to compiled into the app.
		*/
		if(!m_pImpl->bFromMemory)
		{

			for(i = 0; i < nNumCamera; ++i)
				FREE(pCamera[i].pfAnimFOV);
			FREE(pCamera);

			FREE(pLight);

			for(i = 0; i < nNumMaterial; ++i)
			{
				FREE(pMaterial[i].pszName);
				FREE(pMaterial[i].pszEffectFile);
				FREE(pMaterial[i].pszEffectName);
			}
			FREE(pMaterial);

			for(i = 0; i < nNumMesh; ++i) {
				FREE(pMesh[i].sFaces.pData);
				FREE(pMesh[i].pnStripLength);
				if(pMesh[i].pInterleaved)
				{
					FREE(pMesh[i].pInterleaved);
				}
				else
				{
					FREE(pMesh[i].sVertex.pData);
					FREE(pMesh[i].sNormals.pData);
					FREE(pMesh[i].sTangents.pData);
					FREE(pMesh[i].sBinormals.pData);
					for(unsigned int j = 0; j < pMesh[i].nNumUVW; ++j)
						FREE(pMesh[i].psUVW[j].pData);
					FREE(pMesh[i].sVtxColours.pData);
					FREE(pMesh[i].sBoneIdx.pData);
					FREE(pMesh[i].sBoneWeight.pData);
				}
				FREE(pMesh[i].psUVW);
				pMesh[i].sBoneBatches.Release();
			}
			FREE(pMesh);

			for(i = 0; i < nNumNode; ++i) {
				FREE(pNode[i].pszName);
				FREE(pNode[i].pfAnimPosition);
				FREE(pNode[i].pfAnimRotation);
				FREE(pNode[i].pfAnimScale);
				FREE(pNode[i].pfAnimMatrix);
				pNode[i].nAnimFlags = 0;
			}

			FREE(pNode);

			for(i = 0; i < nNumTexture; ++i)
				FREE(pTexture[i].pszName);
			FREE(pTexture);
		}

		// Free the working space used by the implementation
		DestroyImpl();
	}

	memset(this, 0, sizeof(*this));
}

/*!***************************************************************************
 @Function			SetFrame
 @Input				fFrame			Frame number
 @Description		Set the animation frame for which subsequent Get*() calls
					should return data.
*****************************************************************************/
void CPVRTModelPOD::SetFrame(const VERTTYPE fFrame)
{
	if(nNumFrame) {
		/*
			Limit animation frames.

			Example: If there are 100 frames of animation, the highest frame
			number allowed is 98, since that will blend between frames 98 and
			99. (99 being of course the 100th frame.)
		*/
		_ASSERT(fFrame <= f2vt((float)(nNumFrame-1)));
		m_pImpl->nFrame = (int)vt2f(fFrame);
		m_pImpl->fBlend = fFrame - f2vt(m_pImpl->nFrame);
	}
	else
	{
		m_pImpl->fBlend = 0;
		m_pImpl->nFrame = 0;
	}

	m_pImpl->fFrame = fFrame;
}

/*!***************************************************************************
 @Function			GetRotationMatrix
 @Output			mOut			Rotation matrix
 @Input				node			Node to get the rotation matrix from
 @Description		Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
*****************************************************************************/
void CPVRTModelPOD::GetRotationMatrix(
	PVRTMATRIX		&mOut,
	const SPODNode	&node) const
{
	PVRTQUATERNION	q;

	if(node.pfAnimRotation)
	{
		if(node.nAnimFlags & ePODHasRotationAni)
		{
			PVRTMatrixQuaternionSlerp(
				q,
				(PVRTQUATERNION&)node.pfAnimRotation[4*m_pImpl->nFrame],
				(PVRTQUATERNION&)node.pfAnimRotation[4*(m_pImpl->nFrame+1)], m_pImpl->fBlend);
			PVRTMatrixRotationQuaternion(mOut, q);
		}
		else
		{
			PVRTMatrixRotationQuaternion(mOut, *(PVRTQUATERNION*)node.pfAnimRotation);
		}
	}
	else
	{
		PVRTMatrixIdentity(mOut);
	}
}

/*!***************************************************************************
 @Function		GetRotationMatrix
 @Input			node			Node to get the rotation matrix from
 @Returns		Rotation matrix
 @Description	Generates the world matrix for the given Mesh Instance;
				applies the parent's transform too. Uses animation data.
*****************************************************************************/
PVRTMat4 CPVRTModelPOD::GetRotationMatrix(const SPODNode &node) const
{
	PVRTMat4 mOut;
	GetRotationMatrix(mOut,node);
	return mOut;
}

/*!***************************************************************************
 @Function			GetScalingMatrix
 @Output			mOut			Scaling matrix
 @Input				node			Node to get the rotation matrix from
 @Description		Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
*****************************************************************************/
void CPVRTModelPOD::GetScalingMatrix(
	PVRTMATRIX		&mOut,
	const SPODNode	&node) const
{
	PVRTVECTOR3 v;

	if(node.pfAnimScale)
	{
		if(node.nAnimFlags & ePODHasScaleAni)
		{
			PVRTMatrixVec3Lerp(
				v,
				(PVRTVECTOR3&)node.pfAnimScale[7*(m_pImpl->nFrame+0)],
				(PVRTVECTOR3&)node.pfAnimScale[7*(m_pImpl->nFrame+1)], m_pImpl->fBlend);
			PVRTMatrixScaling(mOut, v.x, v.y, v.z);
		}
		else
		{
			PVRTMatrixScaling(mOut, node.pfAnimScale[0], node.pfAnimScale[1], node.pfAnimScale[2]);
		}
	}
	else
	{
		PVRTMatrixIdentity(mOut);
	}
}

/*!***************************************************************************
 @Function		GetScalingMatrix
 @Input			node			Node to get the rotation matrix from
 @Returns		Scaling matrix
 @Description	Generates the world matrix for the given Mesh Instance;
				applies the parent's transform too. Uses animation data.
*****************************************************************************/
PVRTMat4 CPVRTModelPOD::GetScalingMatrix(const SPODNode &node) const
{
	PVRTMat4 mOut;
	GetScalingMatrix(mOut, node);
	return mOut;
}

/*!***************************************************************************
 @Function			GetTranslation
 @Output			V				Translation vector
 @Input				node			Node to get the translation vector from
 @Description		Generates the translation vector for the given Mesh
					Instance. Uses animation data.
*****************************************************************************/
void CPVRTModelPOD::GetTranslation(
	PVRTVECTOR3		&V,
	const SPODNode	&node) const
{
	if(node.pfAnimPosition)
	{
		if(node.nAnimFlags & ePODHasPositionAni)
		{
			PVRTMatrixVec3Lerp(
				V,
				(PVRTVECTOR3&)node.pfAnimPosition[3 * (m_pImpl->nFrame+0)],
				(PVRTVECTOR3&)node.pfAnimPosition[3 * (m_pImpl->nFrame+1)], m_pImpl->fBlend);
		}
		else
		{
			V = *(PVRTVECTOR3*) node.pfAnimPosition;
		}
	}
	else
	{
		_ASSERT(false);
	}
}

/*!***************************************************************************
 @Function		GetTranslation
 @Input			node			Node to get the translation vector from
 @Returns		Translation vector
 @Description	Generates the translation vector for the given Mesh
				Instance. Uses animation data.
*****************************************************************************/
PVRTVec3 CPVRTModelPOD::GetTranslation(const SPODNode &node) const
{
	PVRTVec3 vOut;
	GetTranslation(vOut, node);
	return vOut;
}

/*!***************************************************************************
 @Function			GetTranslationMatrix
 @Output			mOut			Translation matrix
 @Input				node			Node to get the translation matrix from
 @Description		Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
*****************************************************************************/
void CPVRTModelPOD::GetTranslationMatrix(
	PVRTMATRIX		&mOut,
	const SPODNode	&node) const
{
	PVRTVECTOR3 v;

	if(node.pfAnimPosition)
	{
		if(node.nAnimFlags & ePODHasPositionAni)
		{
			PVRTMatrixVec3Lerp(v,
				(PVRTVECTOR3&)node.pfAnimPosition[3*(m_pImpl->nFrame+0)],
				(PVRTVECTOR3&)node.pfAnimPosition[3*(m_pImpl->nFrame+1)], m_pImpl->fBlend);
			PVRTMatrixTranslation(mOut, v.x, v.y, v.z);
		}
		else
		{
			PVRTMatrixTranslation(mOut, node.pfAnimPosition[0], node.pfAnimPosition[1], node.pfAnimPosition[2]);
		}
	}
	else
	{
		PVRTMatrixIdentity(mOut);
	}
}

/*!***************************************************************************
 @Function		GetTranslationMatrix
 @Input			node			Node to get the translation matrix from
 @Returns		Translation matrix
 @Description	Generates the world matrix for the given Mesh Instance;
				applies the parent's transform too. Uses animation data.
*****************************************************************************/
PVRTMat4 CPVRTModelPOD::GetTranslationMatrix(const SPODNode &node) const
{
	PVRTMat4 mOut;
	GetTranslationMatrix(mOut, node);
	return mOut;
}

/*!***************************************************************************
 @Function		GetTransformationMatrix
 @Output		mOut			Transformation matrix
 @Input			node			Node to get the transformation matrix from
 @Description	Generates the world matrix for the given Mesh Instance;
				applies the parent's transform too. Uses animation data.
*****************************************************************************/
void CPVRTModelPOD::GetTransformationMatrix(PVRTMATRIX &mOut, const SPODNode &node) const
{
	if(node.pfAnimMatrix)
	{
		if(node.nAnimFlags & ePODHasMatrixAni)
		{
			mOut = *((PVRTMATRIX*) &node.pfAnimMatrix[16*m_pImpl->nFrame]);
		}
		else
		{
			mOut = *((PVRTMATRIX*) node.pfAnimMatrix);
		}
	}
	else
	{
		PVRTMatrixIdentity(mOut);
	}
}
/*!***************************************************************************
 @Function			GetWorldMatrixNoCache
 @Output			mOut			World matrix
 @Input				node			Node to get the world matrix from
 @Description		Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
*****************************************************************************/
void CPVRTModelPOD::GetWorldMatrixNoCache(
	PVRTMATRIX		&mOut,
	const SPODNode	&node) const
{
	PVRTMATRIX mTmp;

	if(node.pfAnimMatrix) // The transformations are stored as matrices
		GetTransformationMatrix(mOut, node);
	else
	{
		// Scale
		GetScalingMatrix(mOut, node);

		// Rotation
		GetRotationMatrix(mTmp, node);
		PVRTMatrixMultiply(mOut, mOut, mTmp);

		// Translation
		GetTranslationMatrix(mTmp, node);
		PVRTMatrixMultiply(mOut, mOut, mTmp);
	}

	// Do we have to worry about a parent?
	if(node.nIdxParent < 0)
		return;

	// Apply parent's transform too.
	GetWorldMatrixNoCache(mTmp, pNode[node.nIdxParent]);
	PVRTMatrixMultiply(mOut, mOut, mTmp);
}

/*!***************************************************************************
 @Function		GetWorldMatrixNoCache
 @Input			node			Node to get the world matrix from
 @Returns		World matrix
 @Description	Generates the world matrix for the given Mesh Instance;
				applies the parent's transform too. Uses animation data.
*****************************************************************************/
PVRTMat4 CPVRTModelPOD::GetWorldMatrixNoCache(const SPODNode& node) const
{
	PVRTMat4 mWorld;
	GetWorldMatrixNoCache(mWorld,node);
	return mWorld;
}

/*!***************************************************************************
 @Function			GetWorldMatrix
 @Output			mOut			World matrix
 @Input				node			Node to get the world matrix from
 @Description		Generates the world matrix for the given Mesh Instance;
					applies the parent's transform too. Uses animation data.
*****************************************************************************/
void CPVRTModelPOD::GetWorldMatrix(
	PVRTMATRIX		&mOut,
	const SPODNode	&node) const
{
	unsigned int nIdx;

#ifdef _DEBUG
	++m_pImpl->nWmTotal;
	m_pImpl->fHitPerc = (float)m_pImpl->nWmCacheHit / (float)m_pImpl->nWmTotal;
	m_pImpl->fHitPercZero = (float)m_pImpl->nWmZeroCacheHit / (float)m_pImpl->nWmTotal;
#endif

	// Calculate a node index
	nIdx = (unsigned int)(&node - pNode);

	// There is a dedicated cache for frame 0 data
	if(m_pImpl->fFrame == 0)
	{
		mOut = m_pImpl->pWmZeroCache[nIdx];
#ifdef _DEBUG
		++m_pImpl->nWmZeroCacheHit;
#endif
		return;
	}

	// Has this matrix been calculated & cached?
	if(m_pImpl->fFrame == m_pImpl->pfCache[nIdx])
	{
		mOut = m_pImpl->pWmCache[nIdx];
#ifdef _DEBUG
		++m_pImpl->nWmCacheHit;
#endif
		return;
	}

	GetWorldMatrixNoCache(mOut, node);

	// Cache the matrix
	m_pImpl->pfCache[nIdx]	= m_pImpl->fFrame;
	m_pImpl->pWmCache[nIdx]	= mOut;
}

/*!***************************************************************************
 @Function		GetWorldMatrix
 @Input			node			Node to get the world matrix from
 @Returns		World matrix
 @Description	Generates the world matrix for the given Mesh Instance;
				applies the parent's transform too. Uses animation data.
*****************************************************************************/
PVRTMat4 CPVRTModelPOD::GetWorldMatrix(const SPODNode& node) const
{
	PVRTMat4 mWorld;
	GetWorldMatrix(mWorld,node);
	return mWorld;
}

/*!***************************************************************************
 @Function			GetBoneWorldMatrix
 @Output			mOut			Bone world matrix
 @Input				NodeMesh		Mesh to take the bone matrix from
 @Input				NodeBone		Bone to take the matrix from
 @Description		Generates the world matrix for the given bone.
*****************************************************************************/
void CPVRTModelPOD::GetBoneWorldMatrix(
	PVRTMATRIX		&mOut,
	const SPODNode	&NodeMesh,
	const SPODNode	&NodeBone)
{
	PVRTMATRIX	mTmp;
	VERTTYPE	fFrame;

	fFrame = m_pImpl->fFrame;

	SetFrame(0);

	// Transform by object matrix
	GetWorldMatrix(mOut, NodeMesh);

	// Back transform bone from frame 0 position
	GetWorldMatrix(mTmp, NodeBone);
	PVRTMatrixInverse(mTmp, mTmp);
	PVRTMatrixMultiply(mOut, mOut, mTmp);

	// The bone origin should now be at the origin

	SetFrame(fFrame);

	// Transform bone into frame fFrame position
	GetWorldMatrix(mTmp, NodeBone);
	PVRTMatrixMultiply(mOut, mOut, mTmp);
}

/*!***************************************************************************
 @Function		GetBoneWorldMatrix
 @Input			NodeMesh		Mesh to take the bone matrix from
 @Input			NodeBone		Bone to take the matrix from
 @Returns		Bone world matrix
 @Description	Generates the world matrix for the given bone.
*****************************************************************************/
PVRTMat4 CPVRTModelPOD::GetBoneWorldMatrix(
	const SPODNode	&NodeMesh,
	const SPODNode	&NodeBone)
{
	PVRTMat4 mOut;
	GetBoneWorldMatrix(mOut,NodeMesh,NodeBone);
	return mOut;
}

/*!***************************************************************************
 @Function			GetCamera
 @Output			vFrom			Position of the camera
 @Output			vTo				Target of the camera
 @Output			vUp				Up direction of the camera
 @Input				nIdx			Camera number
 @Return			Camera horizontal FOV
 @Description		Calculate the From, To and Up vectors for the given
					camera. Uses animation data.
					Note that even if the camera has a target, *pvTo is not
					the position of that target. *pvTo is a position in the
					correct direction of the target, one unit away from the
					camera.
*****************************************************************************/
VERTTYPE CPVRTModelPOD::GetCamera(
	PVRTVECTOR3			&vFrom,
	PVRTVECTOR3			&vTo,
	PVRTVECTOR3			&vUp,
	const unsigned int	nIdx) const
{
	PVRTMATRIX		mTmp;
	VERTTYPE		*pfData;
	SPODCamera		*pCam;
	const SPODNode	*pNd;

	_ASSERT(nIdx < nNumCamera);

	// Camera nodes are after the mesh and light nodes in the array
	pNd = &pNode[nNumMeshNode + nNumLight + nIdx];

	pCam = &pCamera[pNd->nIdx];

	GetWorldMatrix(mTmp, *pNd);

	// View position is 0,0,0,1 transformed by world matrix
	vFrom.x = mTmp.f[12];
	vFrom.y = mTmp.f[13];
	vFrom.z = mTmp.f[14];

	// View direction is 0,-1,0,1 transformed by world matrix
	vTo.x = -mTmp.f[4] + mTmp.f[12];
	vTo.y = -mTmp.f[5] + mTmp.f[13];
	vTo.z = -mTmp.f[6] + mTmp.f[14];

#if defined(BUILD_DX9) || defined(BUILD_DX10)
	/*
		When you rotate the camera from "straight forward" to "straight down", in
		D3D the UP vector will be [0, 0, 1]
	*/
	vUp.x = mTmp.f[ 8];
	vUp.y = mTmp.f[ 9];
	vUp.z = mTmp.f[10];
#endif

#if defined(BUILD_OGL) || defined(BUILD_OGLES) || defined(BUILD_OGLES2)
	/*
		When you rotate the camera from "straight forward" to "straight down", in
		OpenGL the UP vector will be [0, 0, -1]
	*/
	vUp.x = -mTmp.f[ 8];
	vUp.y = -mTmp.f[ 9];
	vUp.z = -mTmp.f[10];
#endif

	/*
		Find & calculate FOV value
	*/
	if(pCam->pfAnimFOV) {
		pfData = &pCam->pfAnimFOV[m_pImpl->nFrame];

		return pfData[0] + m_pImpl->fBlend * (pfData[1] - pfData[0]);
	} else {
		return pCam->fFOV;
	}
}

/*!***************************************************************************
 @Function			GetCameraPos
 @Output			vFrom			Position of the camera
 @Output			vTo				Target of the camera
 @Input				nIdx			Camera number
 @Return			Camera horizontal FOV
 @Description		Calculate the position of the camera and its target. Uses
					animation data.
					If the queried camera does not have a target, *pvTo is
					not changed.
*****************************************************************************/
VERTTYPE CPVRTModelPOD::GetCameraPos(
	PVRTVECTOR3			&vFrom,
	PVRTVECTOR3			&vTo,
	const unsigned int	nIdx) const
{
	PVRTMATRIX		mTmp;
	VERTTYPE		*pfData;
	SPODCamera		*pCam;
	const SPODNode	*pNd;

	_ASSERT(nIdx < nNumCamera);

	// Camera nodes are after the mesh and light nodes in the array
	pNd = &pNode[nNumMeshNode + nNumLight + nIdx];

	// View position is 0,0,0,1 transformed by world matrix
	GetWorldMatrix(mTmp, *pNd);
	vFrom.x = mTmp.f[12];
	vFrom.y = mTmp.f[13];
	vFrom.z = mTmp.f[14];

	pCam = &pCamera[pNd->nIdx];
	if(pCam->nIdxTarget >= 0)
	{
		// View position is 0,0,0,1 transformed by world matrix
		GetWorldMatrix(mTmp, pNode[pCam->nIdxTarget]);
		vTo.x = mTmp.f[12];
		vTo.y = mTmp.f[13];
		vTo.z = mTmp.f[14];
	}

	/*
		Find & calculate FOV value
	*/
	if(pCam->pfAnimFOV) {
		pfData = &pCam->pfAnimFOV[m_pImpl->nFrame];

		return pfData[0] + m_pImpl->fBlend * (pfData[1] - pfData[0]);
	} else {
		return pCam->fFOV;
	}
}

/*!***************************************************************************
 @Function			GetLight
 @Output			vPos			Position of the light
 @Output			vDir			Direction of the light
 @Input				nIdx			Light number
 @Description		Calculate the position and direction of the given Light.
					Uses animation data.
*****************************************************************************/
void CPVRTModelPOD::GetLight(
	PVRTVECTOR3			&vPos,
	PVRTVECTOR3			&vDir,
	const unsigned int	nIdx) const
{
	PVRTMATRIX		mTmp;
	const SPODNode	*pNd;

	_ASSERT(nIdx < nNumLight);

	// Light nodes are after the mesh nodes in the array
	pNd = &pNode[nNumMeshNode + nIdx];

	GetWorldMatrix(mTmp, *pNd);

	// View position is 0,0,0,1 transformed by world matrix
	vPos.x = mTmp.f[12];
	vPos.y = mTmp.f[13];
	vPos.z = mTmp.f[14];

	// View direction is 0,-1,0,0 transformed by world matrix
	vDir.x = -mTmp.f[4];
	vDir.y = -mTmp.f[5];
	vDir.z = -mTmp.f[6];
}

/*!***************************************************************************
 @Function		GetLight
 @Input			nIdx			Light number
 @Return		PVRTVec4 position/direction of light with w set correctly
 @Description	Calculate the position or direction of the given Light.
				Uses animation data.
*****************************************************************************/
PVRTVec4 CPVRTModelPOD::GetLightPosition(const unsigned int u32Idx) const
{	// TODO: make this a real function instead of just wrapping GetLight()
	PVRTVec3 vPos, vDir;
	GetLight(vPos,vDir,u32Idx);

	_ASSERT(u32Idx < nNumLight);
	_ASSERT(pLight[u32Idx].eType!=ePODDirectional);
	return PVRTVec4(vPos,1);
}

/*!***************************************************************************
 @Function		GetLightDirection
 @Input			u32Idx			Light number
 @Return		PVRTVec4 direction of light with w set correctly
 @Description	Calculate the direction of the given Light. Uses animation data.
*****************************************************************************/
PVRTVec4 CPVRTModelPOD::GetLightDirection(const unsigned int u32Idx) const
{	// TODO: make this a real function instead of just wrapping GetLight()
	PVRTVec3 vPos, vDir;
	GetLight(vPos,vDir,u32Idx);

	_ASSERT(u32Idx < nNumLight);
	_ASSERT(pLight[u32Idx].eType!=ePODPoint);
	return PVRTVec4(vDir,0);
}

/*!***************************************************************************
 @Function			CreateSkinIdxWeight
 @Output			pIdx				Four bytes containing matrix indices for vertex (0..255) (D3D: use UBYTE4)
 @Output			pWeight				Four bytes containing blend weights for vertex (0.0 .. 1.0) (D3D: use D3DCOLOR)
 @Input				nVertexBones		Number of bones this vertex uses
 @Input				pnBoneIdx			Pointer to 'nVertexBones' indices
 @Input				pfBoneWeight		Pointer to 'nVertexBones' blend weights
 @Description		Creates the matrix indices and blend weights for a boned
					vertex. Call once per vertex of a boned mesh.
*****************************************************************************/
EPVRTError CPVRTModelPOD::CreateSkinIdxWeight(
	char			* const pIdx,			// Four bytes containing matrix indices for vertex (0..255) (D3D: use UBYTE4)
	char			* const pWeight,		// Four bytes containing blend weights for vertex (0.0 .. 1.0) (D3D: use D3DCOLOR)
	const int		nVertexBones,			// Number of bones this vertex uses
	const int		* const pnBoneIdx,		// Pointer to 'nVertexBones' indices
	const VERTTYPE	* const pfBoneWeight)	// Pointer to 'nVertexBones' blend weights
{
	int i, nSum;
	int nIdx[4];
	int nWeight[4];

	for(i = 0; i < nVertexBones; ++i)
	{
		nIdx[i]		= pnBoneIdx[i];
		nWeight[i]	= (int)vt2f((VERTTYPEMUL(f2vt(255.0f), pfBoneWeight[i])));

		if(nIdx[i] > 255)
		{
			PVRTErrorOutputDebug("Too many bones (highest index is 255).\n");
			return PVR_FAIL;
		}

		nWeight[i]	= PVRT_MAX(nWeight[i], 0);
		nWeight[i]	= PVRT_MIN(nWeight[i], 255);
	}

	for(; i < 4; ++i)
	{
		nIdx[i]		= 0;
		nWeight[i]	= 0;
	}

	if(nVertexBones)
	{
		// It's important the weights sum to 1
		nSum = 0;
		for(i = 0; i < 4; ++i)
			nSum += nWeight[i];

		if(!nSum)
			return PVR_FAIL;

		_ASSERT(nSum <= 255);

		i = 0;
		while(nSum < 255)
		{
			if(nWeight[i]) {
				++nWeight[i];
				++nSum;
			}

			if(++i > 3)
				i = 0;
		}

		_ASSERT(nSum == 255);
	}

#if defined(BUILD_DX9)
	*(unsigned int*)pIdx = D3DCOLOR_ARGB(nIdx[3], nIdx[2], nIdx[1], nIdx[0]);					// UBYTE4 is WZYX
	*(unsigned int*)pWeight = D3DCOLOR_RGBA(nWeight[0], nWeight[1], nWeight[2], nWeight[3]);	// D3DCOLORs are WXYZ
#endif
#if defined(BUILD_DX10)
	*(unsigned int*)pIdx = D3DXCOLOR((float)nIdx[3], (float)nIdx[2],(float) nIdx[1], (float)nIdx[0]);					//
	*(unsigned int*)pWeight = D3DXCOLOR((float)nWeight[0], (float)nWeight[1], (float)nWeight[2], (float)nWeight[3]);	//
#endif

#if defined(BUILD_OGL) || defined(BUILD_OGLES) || defined(BUILD_OGLES2)
	// Return indices and weights as bytes
	for(i = 0; i < 4; ++i)
	{
		pIdx[i]		= nIdx[i];
		pWeight[i]	= nWeight[i];
	}
#endif

	return PVR_SUCCESS;
}

/*!***************************************************************************
 @Function			SavePOD
 @Input				pszFilename		Filename to save to
 @Input				pszExpOpt		A string containing the options used by the exporter
 @Description		Save a binary POD file (.POD).
*****************************************************************************/
EPVRTError CPVRTModelPOD::SavePOD(const char * const pszFilename, const char * const pszExpOpt, const char * const pszHistory)
{
	FILE	*pFile;
	bool	bRet;

	pFile = fopen(pszFilename, "wb+");
	if(!pFile)
		return PVR_FAIL;

	bRet = WritePOD(pFile, pszExpOpt, pszHistory, *this);

	// Done
	fclose(pFile);
	return bRet ? PVR_SUCCESS : PVR_FAIL;
}

/****************************************************************************
** Code
****************************************************************************/

/*!***************************************************************************
 @Function			PVRTModelPODDataTypeSize
 @Input				type		Type to get the size of
 @Return			Size of the data element
 @Description		Returns the size of each data element.
*****************************************************************************/
size_t PVRTModelPODDataTypeSize(const EPVRTDataType type)
{
	switch(type)
	{
	default:
		_ASSERT(false);
		return 0;
	case EPODDataFloat:
		return sizeof(float);
	case EPODDataInt:
		return sizeof(int);
	case EPODDataShort:
	case EPODDataShortNorm:
	case EPODDataUnsignedShort:
		return sizeof(unsigned short);
	case EPODDataRGBA:
		return sizeof(unsigned int);
	case EPODDataARGB:
		return sizeof(unsigned int);
	case EPODDataD3DCOLOR:
		return sizeof(unsigned int);
	case EPODDataUBYTE4:
		return sizeof(unsigned int);
	case EPODDataDEC3N:
		return sizeof(unsigned int);
	case EPODDataFixed16_16:
		return sizeof(unsigned int);
	case EPODDataUnsignedByte:
	case EPODDataByte:
	case EPODDataByteNorm:
		return sizeof(unsigned char);
	}
}

/*!***************************************************************************
@Function			PVRTModelPODDataTypeComponentCount
@Input				type		Type to get the number of components from
@Return				number of components in the data element
@Description		Returns the number of components in a data element.
*****************************************************************************/
size_t PVRTModelPODDataTypeComponentCount(const EPVRTDataType type)
{
	switch(type)
	{
	default:
		_ASSERT(false);
		return 0;

	case EPODDataFloat:
	case EPODDataInt:
	case EPODDataShort:
	case EPODDataShortNorm:
	case EPODDataUnsignedShort:
	case EPODDataFixed16_16:
	case EPODDataByte:
	case EPODDataByteNorm:
	case EPODDataUnsignedByte:
		return 1;

	case EPODDataDEC3N:
		return 3;

	case EPODDataRGBA:
	case EPODDataARGB:
	case EPODDataD3DCOLOR:
	case EPODDataUBYTE4:
		return 4;
	}
}

/*!***************************************************************************
 @Function			PVRTModelPODDataStride
 @Input				data		Data elements
 @Return			Size of the vector elements
 @Description		Returns the size of the vector of data elements.
*****************************************************************************/
size_t PVRTModelPODDataStride(const CPODData &data)
{
	return PVRTModelPODDataTypeSize(data.eType) * data.n;
}

/*!***************************************************************************
 @Function			PVRTModelPODDataConvert
 @Modified			data		Data elements to convert
 @Input				eNewType	New type of elements
 @Input				nCnt		Number of elements
 @Description		Convert the format of the array of vectors.
*****************************************************************************/
void PVRTModelPODDataConvert(CPODData &data, const unsigned int nCnt, const EPVRTDataType eNewType)
{
	PVRTVECTOR4f	v;
	unsigned int	i;
	CPODData		old;

	if(!data.pData || data.eType == eNewType)
		return;

	old = data;

	switch(eNewType)
	{
	case EPODDataFloat:
	case EPODDataInt:
	case EPODDataUnsignedShort:
	case EPODDataFixed16_16:
	case EPODDataUnsignedByte:
	case EPODDataShort:
	case EPODDataShortNorm:
	case EPODDataByte:
	case EPODDataByteNorm:
		data.n = old.n * PVRTModelPODDataTypeComponentCount(old.eType);
		break;
	case EPODDataRGBA:
	case EPODDataARGB:
	case EPODDataD3DCOLOR:
	case EPODDataUBYTE4:
	case EPODDataDEC3N:
		data.n = 1;
		break;
	default:
		break;
	}

	data.eType = eNewType;
	data.nStride = (unsigned int)PVRTModelPODDataStride(data);

	// If the old & new strides are identical, we can convert it in place
	if(old.nStride != data.nStride)
	{
		data.pData = (unsigned char*)malloc(data.nStride * nCnt);
	}

	for(i = 0; i < nCnt; ++i)
	{
		PVRTVertexRead(&v, old.pData + i * old.nStride, old.eType, old.n);
		PVRTVertexWrite(data.pData + i * data.nStride, eNewType, data.n * PVRTModelPODDataTypeComponentCount(data.eType), &v);
	}

	if(old.nStride != data.nStride)
	{
		FREE(old.pData);
	}
}

static int BitCount(unsigned int n)
{
	int nRet = 0;
	while(n)
	{
		if(n & 0x01)
			++nRet;

		n >>= 1;
	}
	return nRet;
}

/*!***************************************************************************
 @Function			PVRTModelPODDataShred
 @Modified			data		Data elements to modify
 @Input				nCnt		Number of elements
 @Input				nMask		Channel masks
 @Description		Reduce the number of dimensions in 'data' using the channel
					masks in 'nMask'.
*****************************************************************************/
void PVRTModelPODDataShred(CPODData &data, const unsigned int nCnt, const unsigned int nMask)
{
	CPODData		old;
	PVRTVECTOR4f	v;
	unsigned int	i, j, nCh;

	if(!data.pData)
		return;

	old = data;

	// Count the number of output channels
	data.n = BitCount(nMask);
	if(data.n > old.n)
		data.n = old.n;

	// Allocate output memory
	data.nStride = (unsigned int)PVRTModelPODDataStride(data);

	if(data.nStride == 0)
	{
		FREE(data.pData);
		return;
	}

	data.pData = (unsigned char*)malloc(data.nStride * nCnt);

	for(i = 0; i < nCnt; ++i)
	{
		// Read the vector
		PVRTVertexRead(&v, old.pData + i * old.nStride, old.eType, old.n);

		// Shred the vector
		nCh = 0;
		for(j = 0; j < 4; ++j)
		{
			if(nMask & (1 << j))
			{
				((unsigned int*)&v)[nCh] = ((unsigned int*)&v)[j];
				++nCh;
			}
		}

		for(; nCh < 4; ++nCh)
			((unsigned int*)&v)[nCh] = 0;

		// Write the vector
		PVRTVertexWrite((char*)data.pData + i * data.nStride, data.eType, data.n * PVRTModelPODDataTypeComponentCount(data.eType), &v);
	}

	FREE(old.pData);
}

/*!***************************************************************************
 @Function			InterleaveArray
 @Modified			pInterleaved
 @Modified			data
 @Input				nNumVertex
 @Input				nStride
 @Input				nOffset
 @Description		Interleaves the pod data
*****************************************************************************/
static void InterleaveArray(
	char			* const pInterleaved,
	CPODData		&data,
	const int		nNumVertex,
	const size_t	nStride,
	size_t			&nOffset)
{
	if(!data.nStride)
		return;

	for(int i = 0; i < nNumVertex; ++i)
		memcpy(pInterleaved + i * nStride + nOffset, (char*)data.pData + i * data.nStride, data.nStride);
	FREE(data.pData);
	data.nStride	= (unsigned int)nStride;
	data.pData		= (unsigned char*)nOffset;
	nOffset += (int)PVRTModelPODDataStride(data);
}

/*!***************************************************************************
 @Function			DeinterleaveArray
 @Input				data
 @Input				pInter
 @Input				nNumVertex
 @Description		DeInterleaves the pod data
*****************************************************************************/
static void DeinterleaveArray(
	CPODData			&data,
	const void			* const pInter,
	const int			nNumVertex)
{
	unsigned int	nSrcStride	= data.nStride;
	unsigned int	nDestStride	= (unsigned int)PVRTModelPODDataStride(data);
	const char		*pSrc		= (char*)pInter + (size_t)data.pData;

	if(!nSrcStride)
		return;

	data.pData = 0;
	SafeAlloc(data.pData, nDestStride * nNumVertex);
	data.nStride	= nDestStride;

	for(int i = 0; i < nNumVertex; ++i)
		memcpy((char*)data.pData + i * nDestStride, pSrc + i * nSrcStride, nDestStride);
}

/*!***************************************************************************
 @Function		PVRTModelPODToggleInterleaved
 @Modified		mesh		Mesh to modify
 @Description	Switches the supplied mesh to or from interleaved data format.
*****************************************************************************/
void PVRTModelPODToggleInterleaved(SPODMesh &mesh)
{
	unsigned int i;

	if(!mesh.nNumVertex)
		return;

	if(mesh.pInterleaved)
	{
		/*
			De-interleave
		*/
		DeinterleaveArray(mesh.sVertex, mesh.pInterleaved, mesh.nNumVertex);
		DeinterleaveArray(mesh.sNormals, mesh.pInterleaved, mesh.nNumVertex);
		DeinterleaveArray(mesh.sTangents, mesh.pInterleaved, mesh.nNumVertex);
		DeinterleaveArray(mesh.sBinormals, mesh.pInterleaved, mesh.nNumVertex);

		for(i = 0; i < mesh.nNumUVW; ++i)
			DeinterleaveArray(mesh.psUVW[i], mesh.pInterleaved, mesh.nNumVertex);

		DeinterleaveArray(mesh.sVtxColours, mesh.pInterleaved, mesh.nNumVertex);
		DeinterleaveArray(mesh.sBoneIdx, mesh.pInterleaved, mesh.nNumVertex);
		DeinterleaveArray(mesh.sBoneWeight, mesh.pInterleaved, mesh.nNumVertex);
		FREE(mesh.pInterleaved);
	}
	else
	{
		size_t nStride, nOffset;

		/*
			Interleave
		*/

		// Calculate how much data the interleaved array must store
		nStride = PVRTModelPODDataStride(mesh.sVertex);
		nStride += PVRTModelPODDataStride(mesh.sNormals);
		nStride += PVRTModelPODDataStride(mesh.sTangents);
		nStride += PVRTModelPODDataStride(mesh.sBinormals);

		for(i = 0; i < mesh.nNumUVW; ++i)
			nStride += PVRTModelPODDataStride(mesh.psUVW[i]);

		nStride += PVRTModelPODDataStride(mesh.sVtxColours);
		nStride += PVRTModelPODDataStride(mesh.sBoneIdx);
		nStride += PVRTModelPODDataStride(mesh.sBoneWeight);

		// Allocate interleaved array
		SafeAlloc(mesh.pInterleaved, mesh.nNumVertex * nStride);

		// Interleave the data
		nOffset = 0;
		InterleaveArray((char*)mesh.pInterleaved, mesh.sVertex, mesh.nNumVertex, nStride, nOffset);
		InterleaveArray((char*)mesh.pInterleaved, mesh.sNormals, mesh.nNumVertex, nStride, nOffset);
		InterleaveArray((char*)mesh.pInterleaved, mesh.sTangents, mesh.nNumVertex, nStride, nOffset);
		InterleaveArray((char*)mesh.pInterleaved, mesh.sBinormals, mesh.nNumVertex, nStride, nOffset);
		InterleaveArray((char*)mesh.pInterleaved, mesh.sVtxColours, mesh.nNumVertex, nStride, nOffset);

		for(i = 0; i < mesh.nNumUVW; ++i)
			InterleaveArray((char*)mesh.pInterleaved, mesh.psUVW[i], mesh.nNumVertex, nStride, nOffset);

		InterleaveArray((char*)mesh.pInterleaved, mesh.sBoneIdx, mesh.nNumVertex, nStride, nOffset);
		InterleaveArray((char*)mesh.pInterleaved, mesh.sBoneWeight, mesh.nNumVertex, nStride, nOffset);
	}
}

/*!***************************************************************************
 @Function			PVRTModelPODDeIndex
 @Modified			mesh		Mesh to modify
 @Description		De-indexes the supplied mesh. The mesh must be
					Interleaved before calling this function.
*****************************************************************************/
void PVRTModelPODDeIndex(SPODMesh &mesh)
{
	unsigned char *pNew = 0;

	if(!mesh.pInterleaved || !mesh.nNumVertex)
		return;

	_ASSERT(mesh.nNumVertex && mesh.nNumFaces);

	// Create a new vertex list
	mesh.nNumVertex = PVRTModelPODCountIndices(mesh);
	SafeAlloc(pNew, mesh.sVertex.nStride * mesh.nNumVertex);

	// Deindex the vertices
	for(unsigned int i = 0; i < mesh.nNumVertex; ++i)
		memcpy(pNew + i * mesh.sVertex.nStride, (char*)mesh.pInterleaved + ((unsigned short*)mesh.sFaces.pData)[i] * mesh.sVertex.nStride, mesh.sVertex.nStride);

	// Replace the old vertex list
	FREE(mesh.pInterleaved);
	mesh.pInterleaved = pNew;

	// Get rid of the index list
	FREE(mesh.sFaces.pData);
	mesh.sFaces.n		= 0;
	mesh.sFaces.nStride	= 0;
}

/*!***************************************************************************
 @Function			PVRTModelPODToggleStrips
 @Modified			mesh		Mesh to modify
 @Description		Converts the supplied mesh to or from strips.
*****************************************************************************/
void PVRTModelPODToggleStrips(SPODMesh &mesh)
{
	CPODData	old;
	size_t	nIdxSize, nTriStride;

	if(!mesh.nNumFaces)
		return;

	_ASSERT(mesh.sFaces.n == 1);
	nIdxSize	= PVRTModelPODDataTypeSize(mesh.sFaces.eType);
	nTriStride	= PVRTModelPODDataStride(mesh.sFaces) * 3;

	old					= mesh.sFaces;
	mesh.sFaces.pData	= 0;
	SafeAlloc(mesh.sFaces.pData, nTriStride * mesh.nNumFaces);

	if(mesh.nNumStrips)
	{
		unsigned int nListIdxCnt, nStripIdxCnt;

		//	Convert to list
		nListIdxCnt		= 0;
		nStripIdxCnt	= 0;

		for(unsigned int i = 0; i < mesh.nNumStrips; ++i)
		{
			for(unsigned int j = 0; j < mesh.pnStripLength[i]; ++j)
			{
				if(j)
				{
					_ASSERT(j == 1); // Because this will surely break with any other number

					memcpy(
						(char*)mesh.sFaces.pData	+ nIdxSize * nListIdxCnt,
						(char*)old.pData			+ nIdxSize * (nStripIdxCnt - 1),
						nIdxSize);
					nListIdxCnt += 1;

					memcpy(
						(char*)mesh.sFaces.pData	+ nIdxSize * nListIdxCnt,
						(char*)old.pData			+ nIdxSize * (nStripIdxCnt - 2),
						nIdxSize);
					nListIdxCnt += 1;

					memcpy(
						(char*)mesh.sFaces.pData	+ nIdxSize * nListIdxCnt,
						(char*)old.pData			+ nIdxSize * nStripIdxCnt,
						nIdxSize);
					nListIdxCnt += 1;

					nStripIdxCnt += 1;
				}
				else
				{
					memcpy(
						(char*)mesh.sFaces.pData	+ nIdxSize * nListIdxCnt,
						(char*)old.pData			+ nIdxSize * nStripIdxCnt,
						nTriStride);

					nStripIdxCnt += 3;
					nListIdxCnt += 3;
				}
			}
		}

		_ASSERT(nListIdxCnt == mesh.nNumFaces*3);
		FREE(mesh.pnStripLength);
		mesh.nNumStrips = 0;
	}
	else
	{
		int		nIdxCnt;
		int		nBatchCnt;
		unsigned int n0, n1, n2;
		unsigned int p0, p1, p2, nFaces;
		unsigned char* pFaces;

		//	Convert to strips
		mesh.pnStripLength	= (unsigned int*)calloc(mesh.nNumFaces, sizeof(*mesh.pnStripLength));
		mesh.nNumStrips		= 0;
		nIdxCnt				= 0;
		nBatchCnt			= mesh.sBoneBatches.nBatchCnt ? mesh.sBoneBatches.nBatchCnt : 1;

		for(int h = 0; h < nBatchCnt; ++h)
		{
			n0 = 0;
			n1 = 0;
			n2 = 0;

			if(!mesh.sBoneBatches.nBatchCnt)
			{
				nFaces = mesh.nNumFaces;
				pFaces = old.pData;
			}
			else
			{
				if(h + 1 < mesh.sBoneBatches.nBatchCnt)
					nFaces = mesh.sBoneBatches.pnBatchOffset[h+1] - mesh.sBoneBatches.pnBatchOffset[h];
				else
					nFaces = mesh.nNumFaces - mesh.sBoneBatches.pnBatchOffset[h];

				pFaces = &old.pData[3 * mesh.sBoneBatches.pnBatchOffset[h] * old.nStride];
			}

			for(unsigned int i = 0; i < nFaces; ++i)
			{
				p0 = n0;
				p1 = n1;
				p2 = n2;

				PVRTVertexRead(&n0, (char*)pFaces + (3 * i + 0) * old.nStride, old.eType);
				PVRTVertexRead(&n1, (char*)pFaces + (3 * i + 1) * old.nStride, old.eType);
				PVRTVertexRead(&n2, (char*)pFaces + (3 * i + 2) * old.nStride, old.eType);

				if(mesh.pnStripLength[mesh.nNumStrips])
				{
					if(mesh.pnStripLength[mesh.nNumStrips] & 0x01)
					{
						if(p1 == n1 && p2 == n0)
						{
							PVRTVertexWrite((char*)mesh.sFaces.pData + nIdxCnt * mesh.sFaces.nStride, mesh.sFaces.eType, n2);
							++nIdxCnt;
							mesh.pnStripLength[mesh.nNumStrips] += 1;
							continue;
						}
					}
					else
					{
						if(p2 == n1 && p0 == n0)
						{
							PVRTVertexWrite((char*)mesh.sFaces.pData + nIdxCnt * mesh.sFaces.nStride, mesh.sFaces.eType, n2);
							++nIdxCnt;
							mesh.pnStripLength[mesh.nNumStrips] += 1;
							continue;
						}
					}

					++mesh.nNumStrips;
				}

				//	Start of strip, copy entire triangle
				PVRTVertexWrite((char*)mesh.sFaces.pData + nIdxCnt * mesh.sFaces.nStride, mesh.sFaces.eType, n0);
				++nIdxCnt;
				PVRTVertexWrite((char*)mesh.sFaces.pData + nIdxCnt * mesh.sFaces.nStride, mesh.sFaces.eType, n1);
				++nIdxCnt;
				PVRTVertexWrite((char*)mesh.sFaces.pData + nIdxCnt * mesh.sFaces.nStride, mesh.sFaces.eType, n2);
				++nIdxCnt;

				mesh.pnStripLength[mesh.nNumStrips] += 1;
			}
		}

		if(mesh.pnStripLength[mesh.nNumStrips])
			++mesh.nNumStrips;

		SafeRealloc(mesh.sFaces.pData, nIdxCnt * nIdxSize);
		mesh.pnStripLength	= (unsigned int*)realloc(mesh.pnStripLength, sizeof(*mesh.pnStripLength) * mesh.nNumStrips);
	}

	FREE(old.pData);
}

/*!***************************************************************************
 @Function		PVRTModelPODCountIndices
 @Input			mesh		Mesh
 @Return		Number of indices used by mesh
 @Description	Counts the number of indices of a mesh
*****************************************************************************/
unsigned int PVRTModelPODCountIndices(const SPODMesh &mesh)
{
	if(mesh.nNumStrips)
	{
		unsigned int i, n = 0;

		for(i = 0; i < mesh.nNumStrips; ++i)
			n += mesh.pnStripLength[i] + 2;

		return n;
	}

	return mesh.nNumFaces * 3;
}

static void FloatToFixed(int * const pn, const float * const pf, unsigned int n)
{
	if(!pn || !pf) return;
	while(n)
	{
		--n;
		pn[n] = (int)(pf[n] * (float)(1<<16));
	}
}
static void FixedToFloat(float * const pf, const int * const pn, unsigned int n)
{
	if(!pn || !pf) return;
	while(n)
	{
		--n;
		pf[n] = (float)pn[n] / (float)(1<<16);
	}
}

/*!***************************************************************************
 @Function		PVRTModelPODToggleFixedPoint
 @Modified		s		Scene to modify
 @Description	Switch all non-vertex data between fixed-point and
				floating-point.
*****************************************************************************/
void PVRTModelPODToggleFixedPoint(SPODScene &s)
{
	unsigned int i;
	int i32TransformNo;

	if(s.nFlags & PVRTMODELPODSF_FIXED)
	{
		/*
			Convert to floating-point
		*/
		for(i = 0; i < s.nNumCamera; ++i)
		{
			FixedToFloat((float*)&s.pCamera[i].fFOV, (int*)&s.pCamera[i].fFOV, 1);
			FixedToFloat((float*)&s.pCamera[i].fFar, (int*)&s.pCamera[i].fFar, 1);
			FixedToFloat((float*)&s.pCamera[i].fNear, (int*)&s.pCamera[i].fNear, 1);
			FixedToFloat((float*)s.pCamera[i].pfAnimFOV, (int*)s.pCamera[i].pfAnimFOV, s.nNumFrame);
		}

		for(i = 0; i < s.nNumLight; ++i)
		{
			FixedToFloat((float*)&s.pLight[i].pfColour, (int*)&s.pLight[i].pfColour, 3);
			FixedToFloat((float*)&s.pLight[i].fConstantAttenuation, (int*)&s.pLight[i].fConstantAttenuation, 1);
			FixedToFloat((float*)&s.pLight[i].fLinearAttenuation,	(int*)&s.pLight[i].fLinearAttenuation, 1);
			FixedToFloat((float*)&s.pLight[i].fQuadraticAttenuation,(int*)&s.pLight[i].fQuadraticAttenuation, 1);
			FixedToFloat((float*)&s.pLight[i].fFalloffAngle,		(int*)&s.pLight[i].fFalloffAngle, 1);
			FixedToFloat((float*)&s.pLight[i].fFalloffExponent,		(int*)&s.pLight[i].fFalloffExponent, 1);
		}

		for(i = 0; i < s.nNumNode; ++i)
		{
			i32TransformNo = s.pNode[i].nAnimFlags & ePODHasPositionAni ? s.nNumFrame : 1;
			FixedToFloat((float*)s.pNode[i].pfAnimPosition,	(int*)s.pNode[i].pfAnimPosition,	3  * i32TransformNo);

			i32TransformNo = s.pNode[i].nAnimFlags & ePODHasRotationAni ? s.nNumFrame : 1;
			FixedToFloat((float*)s.pNode[i].pfAnimRotation,	(int*)s.pNode[i].pfAnimRotation,	4  * i32TransformNo);

			i32TransformNo = s.pNode[i].nAnimFlags & ePODHasScaleAni ? s.nNumFrame : 1;
			FixedToFloat((float*)s.pNode[i].pfAnimScale,	(int*)s.pNode[i].pfAnimScale,		7  * i32TransformNo);

			i32TransformNo = s.pNode[i].nAnimFlags & ePODHasMatrixAni ? s.nNumFrame : 1;
			FixedToFloat((float*)s.pNode[i].pfAnimMatrix,	(int*)s.pNode[i].pfAnimMatrix,		16 * i32TransformNo);
		}

		for(i = 0; i < s.nNumMaterial; ++i)
		{
			FixedToFloat((float*)&s.pMaterial[i].fMatOpacity,	(int*)&s.pMaterial[i].fMatOpacity,		1);
			FixedToFloat((float*)s.pMaterial[i].pfMatAmbient,	(int*)s.pMaterial[i].pfMatAmbient,		3);
			FixedToFloat((float*)s.pMaterial[i].pfMatDiffuse,	(int*)s.pMaterial[i].pfMatDiffuse,		3);
			FixedToFloat((float*)s.pMaterial[i].pfMatSpecular,	(int*)s.pMaterial[i].pfMatSpecular,		3);
			FixedToFloat((float*)&s.pMaterial[i].fMatShininess,	(int*)&s.pMaterial[i].fMatShininess,	1);
		}

		FixedToFloat((float*)s.pfColourBackground,	(int*)s.pfColourBackground,	3);
		FixedToFloat((float*)s.pfColourAmbient,		(int*)s.pfColourAmbient,	3);
	}
	else
	{
		/*
			Convert to Fixed-point
		*/
		for(i = 0; i < s.nNumCamera; ++i)
		{
			FloatToFixed((int*)&s.pCamera[i].fFOV, (float*)&s.pCamera[i].fFOV, 1);
			FloatToFixed((int*)&s.pCamera[i].fFar, (float*)&s.pCamera[i].fFar, 1);
			FloatToFixed((int*)&s.pCamera[i].fNear, (float*)&s.pCamera[i].fNear, 1);
			FloatToFixed((int*)s.pCamera[i].pfAnimFOV, (float*)s.pCamera[i].pfAnimFOV, s.nNumFrame);
		}

		for(i = 0; i < s.nNumLight; ++i)
		{
			FloatToFixed((int*)&s.pLight[i].pfColour, (float*)&s.pLight[i].pfColour, 3);
			FloatToFixed((int*)&s.pLight[i].fConstantAttenuation, (float*)&s.pLight[i].fConstantAttenuation, 1);
			FloatToFixed((int*)&s.pLight[i].fLinearAttenuation,	(float*)&s.pLight[i].fLinearAttenuation, 1);
			FloatToFixed((int*)&s.pLight[i].fQuadraticAttenuation,(float*)&s.pLight[i].fQuadraticAttenuation, 1);
			FloatToFixed((int*)&s.pLight[i].fFalloffAngle,		(float*)&s.pLight[i].fFalloffAngle, 1);
			FloatToFixed((int*)&s.pLight[i].fFalloffExponent,		(float*)&s.pLight[i].fFalloffExponent, 1);
		}

		for(i = 0; i < s.nNumNode; ++i)
		{
			i32TransformNo = s.pNode[i].nAnimFlags & ePODHasPositionAni ? s.nNumFrame : 1;
			FloatToFixed((int*)s.pNode[i].pfAnimPosition,	(float*)s.pNode[i].pfAnimPosition,	3 * i32TransformNo);

			i32TransformNo = s.pNode[i].nAnimFlags & ePODHasRotationAni ? s.nNumFrame : 1;
			FloatToFixed((int*)s.pNode[i].pfAnimRotation,	(float*)s.pNode[i].pfAnimRotation,	4 * i32TransformNo);

			i32TransformNo = s.pNode[i].nAnimFlags & ePODHasScaleAni ? s.nNumFrame : 1;
			FloatToFixed((int*)s.pNode[i].pfAnimScale,	(float*)s.pNode[i].pfAnimScale,		7 * i32TransformNo);

			i32TransformNo = s.pNode[i].nAnimFlags & ePODHasMatrixAni ? s.nNumFrame : 1;
			FloatToFixed((int*)s.pNode[i].pfAnimMatrix,	(float*)s.pNode[i].pfAnimMatrix,		16 * i32TransformNo);
		}

		for(i = 0; i < s.nNumMaterial; ++i)
		{
			FloatToFixed((int*)&s.pMaterial[i].fMatOpacity,	(float*)&s.pMaterial[i].fMatOpacity,		1);
			FloatToFixed((int*)s.pMaterial[i].pfMatAmbient,	(float*)s.pMaterial[i].pfMatAmbient,		3);
			FloatToFixed((int*)s.pMaterial[i].pfMatDiffuse,	(float*)s.pMaterial[i].pfMatDiffuse,		3);
			FloatToFixed((int*)s.pMaterial[i].pfMatSpecular,	(float*)s.pMaterial[i].pfMatSpecular,		3);
			FloatToFixed((int*)&s.pMaterial[i].fMatShininess,	(float*)&s.pMaterial[i].fMatShininess,	1);
		}

		FloatToFixed((int*)s.pfColourBackground,	(float*)s.pfColourBackground,	3);
		FloatToFixed((int*)s.pfColourAmbient,		(float*)s.pfColourAmbient,	3);
	}

	// Done
	s.nFlags ^= PVRTMODELPODSF_FIXED;
}

/*****************************************************************************
 End of file (PVRTModelPOD.cpp)
*****************************************************************************/
