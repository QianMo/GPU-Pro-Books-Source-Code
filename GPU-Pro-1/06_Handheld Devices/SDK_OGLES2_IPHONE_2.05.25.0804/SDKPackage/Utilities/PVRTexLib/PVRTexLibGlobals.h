/******************************************************************************

 @File         PVRTexLibGlobals.h

 @Title        Console Log

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI

 @Description  Macros, enums and constants for PVRTexLib.

******************************************************************************/
#ifndef PVRTEXLIBGLOBALS_H
#define PVRTEXLIBGLOBALS_H

/*****************************************************************************
* Includes
*****************************************************************************/

namespace pvrtexlib
{
	
#ifdef __APPLE__
	/* The classes below are exported */
#pragma GCC visibility push(default)
#endif	

	// DLL defines. Define _WINDLL_IMPORT to use the PVRTexLib dll under Windows

#ifndef PVR_DLL
#ifdef _WINDLL_EXPORT
#define PVR_DLL __declspec(dllexport)
#elif _WINDLL_IMPORT
#define PVR_DLL __declspec(dllimport)
#else
#define PVR_DLL
#endif
#endif


	/*****************************************************************************
	* PVRTexLib Version
	*****************************************************************************/

const int TEXTOOLVERSION_MAJOR = 3;
const int TEXTOOLVERSION_MINOR = 10;
const char* const  TEXTOOLVERSION_STRING = "";

	/*****************************************************************************
	* Arithmetic Macros
	*****************************************************************************/

#define _CLAMP_(X,Xmin,Xmax) (  (X)<(Xmax) ?  (  (X)<(Xmin)?(Xmin):(X)  )  : (Xmax)    )

	/*****************************************************************************
	* Memory Macros
	*****************************************************************************/
#ifndef PVRFREE 
#define PVRFREE(A) { if(A) {free(A); A=NULL;} } 
#endif
#ifndef PVRDELETE 
#define PVRDELETE(A) { if(A) {delete(A); A=NULL;}} 
#endif
#ifndef	PVRDELETEARRAY 
#define PVRDELETEARRAY(A) { if(A) {delete[](A); A=NULL;}} 
#endif

// This macro is used to check at compile time that types are of a certain size
// If the size does not equal the expected size, this typedefs an array of size 0
// which causes a compile error
#define PVRTSIZEASSERT(T, size) typedef int (sizeof_##T)[sizeof(T) == (size)]

	/*****************************************************************************
	* typedefs for standard pixel type channels
	*****************************************************************************/
	typedef		signed char			int8;
	typedef		signed short		int16;
	typedef		signed int			int32;
	typedef		signed long long    int64;
	typedef 	unsigned char		uint8;
	typedef 	unsigned short		uint16;
	typedef 	unsigned int		uint32;
	typedef		unsigned long long	uint64;
	typedef		float				float32;
	typedef		double				float64;

	// check_standard_type is a struct where only the specialisations have a standard declaration (complete type)
	// if this struct is instantiated with a different type then the compiler will choke on it
	// Place a line like: " 		check_standard_type<channelType>();	" in a template function
	// to ensure it won't be called using an non-standard type.
	template<class T> struct check_standard_type;
	template<> struct check_standard_type<uint8> {};
	template<> struct check_standard_type<uint16> {};
	template<> struct check_standard_type<uint32> {};
	template<> struct check_standard_type<float32> {};
	template<> struct check_standard_type<float64> {};

	// simple float16 class
	class float16
	{
		uint16 fVal;
	public:

		float16(){}

		// direct constructor
		template <typename t>
		float16(const t& F)
		{
			fVal = F;
		}

		// conversion to float32 is special
		float16(const float32& F)
		{

			unsigned long uF = *((unsigned long*)&F);
			unsigned long uSign = (uF & 0x10000000) >> 31;
			unsigned long uExponent = (uF & 0x7F800000) >> 23;
			unsigned long uMantissa = (uF & 0x007FFFFF);

			int uExp = uExponent - 127 + 15;
			if(uExp < 1) uExp = 1;
			if(uExp > 30) uExp = 30;
			fVal = (unsigned short)((uSign << 15) |((unsigned int)uExp << 10) | (uMantissa >> 13));
		}

		operator float32() const
		{
			unsigned long uL =	((fVal & 0x1000) << 16) |	//sign
				((  (fVal & 0x7C00) >> 10) - 15 + 127  )  << 23|	//5 bit exponent.
				(fVal & 0x03FF) << 13	;	//10 bit mantissa.
			return *((float*)&uL);
		}
	};

	// check type sizes
	PVRTSIZEASSERT(uint8,1);
	PVRTSIZEASSERT(uint16,2);
	PVRTSIZEASSERT(uint32,4);
	PVRTSIZEASSERT(uint64,8);
	PVRTSIZEASSERT(int8,1);
	PVRTSIZEASSERT(int16,2);
	PVRTSIZEASSERT(int32,4);
	PVRTSIZEASSERT(int64,8);
	PVRTSIZEASSERT(float16,2);
	PVRTSIZEASSERT(float32,4);
	PVRTSIZEASSERT(float64,8);

	// default working type for combining channels (to avoid overflows)
	template<class T> struct combine_type
	{
		typedef float64 type;
	};

	// specialisation for uint32 to avoid overflows 
	template<> struct combine_type<uint32>
	{
		typedef float64 type;
	};

	// specialisation for float32 to avoid overflows 
	template<> struct combine_type<float16>
	{
		typedef float64 type;
	};

	// specialisation for float32 to avoid overflows 
	template<> struct combine_type<float32>
	{
		typedef float64 type;
	};


	// Returns the maximum value before overflow for these types (in the context of texture channels) 
	template<typename tType>
	tType getMaximumValue();

	template<>
	inline float64 getMaximumValue<float64>(){return 1.0;}
	template<>
	inline float32 getMaximumValue<float32>(){return 1.0f;}
	template<>
	inline float16 getMaximumValue<float16>(){return 1.0f;}
	template<>
	inline uint64 getMaximumValue<uint64>(){return 0xffffffffffffffffULL;}
	template<>
	inline uint32 getMaximumValue<uint32>(){return 0xffffffff;}
	template<>
	inline uint16 getMaximumValue<uint16>(){return 0xffff;}
	template<>
	inline uint8 getMaximumValue<uint8>(){return 0xff;}
	template<>
	inline int64 getMaximumValue<int64>(){return 0x7fffffffffffffffLL;}
	template<>
	inline int32 getMaximumValue<int32>(){return 0x7fffffff;}
	template<>
	inline int16 getMaximumValue<int16>(){return 0x7fff;}
	template<>
	inline int8 getMaximumValue<int8>(){return 0x7f;}

	/*****************************************************************************
	* PixelType - corresponds to all pixel formats understood by PVRTexLib
	*****************************************************************************/
	enum PixelType
	{
		MGLPT_ARGB_4444 = 0x00,
		MGLPT_ARGB_1555,
		MGLPT_RGB_565,
		MGLPT_RGB_555,
		MGLPT_RGB_888,
		MGLPT_ARGB_8888,
		MGLPT_ARGB_8332,
		MGLPT_I_8,
		MGLPT_AI_88,
		MGLPT_1_BPP,
		MGLPT_VY1UY0,
		MGLPT_Y1VY0U,
		MGLPT_PVRTC2,
		MGLPT_PVRTC4,

		// OpenGL version of pixel types
		OGL_RGBA_4444= 0x10,
		OGL_RGBA_5551,
		OGL_RGBA_8888,
		OGL_RGB_565,
		OGL_RGB_555,
		OGL_RGB_888,
		OGL_I_8,
		OGL_AI_88,
		OGL_PVRTC2,
		OGL_PVRTC4,
		// OGL_BGRA_8888 extension
		OGL_BGRA_8888,
		OGL_A_8,

		// S3TC
		D3D_DXT1 = 0x20,
		D3D_DXT2,
		D3D_DXT3,
		D3D_DXT4,
		D3D_DXT5,

		D3D_RGB_332,
		D3D_AI_44,
		D3D_LVU_655,
		D3D_XLVU_8888,
		D3D_QWVU_8888,

		//10 bits per channel
		D3D_ABGR_2101010,
		D3D_ARGB_2101010,
		D3D_AWVU_2101010,

		//16 bits per channel
		D3D_GR_1616,
		D3D_VU_1616,
		D3D_ABGR_16161616,

		//HDR formats
		D3D_R16F,
		D3D_GR_1616F,
		D3D_ABGR_16161616F,

		//32 bits per channel
		D3D_R32F,
		D3D_GR_3232F,
		D3D_ABGR_32323232F,

		// Ericsson
		ETC_RGB_4BPP,
		ETC_RGBA_EXPLICIT,				// unimplemented
		ETC_RGBA_INTERPOLATED,			// unimplemented

		// additional pre-DX10
		D3D_A8 = 0x40,
		D3D_V8U8,
		D3D_I16,

		// DX10 


		DX10_R32G32B32A32_FLOAT= 0x50,
		DX10_R32G32B32A32_UINT , 
		DX10_R32G32B32A32_SINT,

		DX10_R32G32B32_FLOAT,
		DX10_R32G32B32_UINT,
		DX10_R32G32B32_SINT,

		DX10_R16G16B16A16_FLOAT ,
		DX10_R16G16B16A16_UNORM,
		DX10_R16G16B16A16_UINT ,
		DX10_R16G16B16A16_SNORM ,
		DX10_R16G16B16A16_SINT ,

		DX10_R32G32_FLOAT ,
		DX10_R32G32_UINT ,
		DX10_R32G32_SINT ,

		DX10_R10G10B10A2_UNORM ,
		DX10_R10G10B10A2_UINT ,

		DX10_R11G11B10_FLOAT ,				// unimplemented

		DX10_R8G8B8A8_UNORM , 
		DX10_R8G8B8A8_UNORM_SRGB ,
		DX10_R8G8B8A8_UINT ,
		DX10_R8G8B8A8_SNORM ,
		DX10_R8G8B8A8_SINT ,

		DX10_R16G16_FLOAT , 
		DX10_R16G16_UNORM , 
		DX10_R16G16_UINT , 
		DX10_R16G16_SNORM ,
		DX10_R16G16_SINT ,

		DX10_R32_FLOAT ,
		DX10_R32_UINT ,
		DX10_R32_SINT ,

		DX10_R8G8_UNORM ,
		DX10_R8G8_UINT ,
		DX10_R8G8_SNORM , 
		DX10_R8G8_SINT ,

		DX10_R16_FLOAT ,
		DX10_R16_UNORM ,
		DX10_R16_UINT ,
		DX10_R16_SNORM ,
		DX10_R16_SINT ,

		DX10_R8_UNORM, 
		DX10_R8_UINT,
		DX10_R8_SNORM,
		DX10_R8_SINT,

		DX10_A8_UNORM, 
		DX10_R1_UNORM, 
		DX10_R9G9B9E5_SHAREDEXP,	// unimplemented
		DX10_R8G8_B8G8_UNORM,		// unimplemented
		DX10_G8R8_G8B8_UNORM,		// unimplemented

		DX10_BC1_UNORM,	
		DX10_BC1_UNORM_SRGB,

		DX10_BC2_UNORM,	
		DX10_BC2_UNORM_SRGB,

		DX10_BC3_UNORM,	
		DX10_BC3_UNORM_SRGB,

		DX10_BC4_UNORM,				// unimplemented
		DX10_BC4_SNORM,				// unimplemented

		DX10_BC5_UNORM,				// unimplemented
		DX10_BC5_SNORM,				// unimplemented

		// OpenVG

		/* RGB{A,X} channel ordering */
		ePT_VG_sRGBX_8888  = 0x90,
		ePT_VG_sRGBA_8888,
		ePT_VG_sRGBA_8888_PRE,
		ePT_VG_sRGB_565,
		ePT_VG_sRGBA_5551,
		ePT_VG_sRGBA_4444,
		ePT_VG_sL_8,
		ePT_VG_lRGBX_8888,
		ePT_VG_lRGBA_8888,
		ePT_VG_lRGBA_8888_PRE,
		ePT_VG_lL_8,
		ePT_VG_A_8,
		ePT_VG_BW_1,

		/* {A,X}RGB channel ordering */
		ePT_VG_sXRGB_8888,
		ePT_VG_sARGB_8888,
		ePT_VG_sARGB_8888_PRE,
		ePT_VG_sARGB_1555,
		ePT_VG_sARGB_4444,
		ePT_VG_lXRGB_8888,
		ePT_VG_lARGB_8888,
		ePT_VG_lARGB_8888_PRE,

		/* BGR{A,X} channel ordering */
		ePT_VG_sBGRX_8888,
		ePT_VG_sBGRA_8888,
		ePT_VG_sBGRA_8888_PRE,
		ePT_VG_sBGR_565,
		ePT_VG_sBGRA_5551,
		ePT_VG_sBGRA_4444,
		ePT_VG_lBGRX_8888,
		ePT_VG_lBGRA_8888,
		ePT_VG_lBGRA_8888_PRE,

		/* {A,X}BGR channel ordering */
		ePT_VG_sXBGR_8888,
		ePT_VG_sABGR_8888 ,
		ePT_VG_sABGR_8888_PRE,
		ePT_VG_sABGR_1555,
		ePT_VG_sABGR_4444,
		ePT_VG_lXBGR_8888,
		ePT_VG_lABGR_8888,
		ePT_VG_lABGR_8888_PRE,

		// max cap for iterating
		END_OF_PIXEL_TYPES,




		MGLPT_NOTYPE = 0xffffffff

	};

	// Standard pixel types for each precision of library
	// These are the formats that the lib actually works on
	const PixelType eInt8StandardPixelType	= DX10_R8G8B8A8_UNORM,
					eInt16StandardPixelType	= D3D_ABGR_16161616,
					eInt32StandardPixelType	= DX10_R32G32B32A32_UINT,
					eFloatStandardPixelType	= D3D_ABGR_32323232F;

	// The number of APIs supported by the library
	// enums for each of these APIs
	enum E_API
	{
		eALL_API=0,
		eOGLES,
		eOGLES2,
		eD3DM,
		eOGL2,
		eDX9,
		eDX10,
		eOVG,
		eMGL,
		NUM_APIS,
	};

	// human readable names for APIs
	const char ppszAPINames[NUM_APIS][16] =
	{
		"All APIs","OpenGL ES 1.x","OpenGL ES 2.0","Direct3D Mobile","OpenGL","DirectX 9","DirectX 10","OpenVG","MGL"
	};

	// names for cube map faces
	const char g_pszCubeFaceNames[6][6] =
	{
		"FRONT","BACK","RIGHT","LEFT","TOP","BASE"
	};

	enum E_PRECMODE
	{	// precision modes - correspond to standard pixel types
		ePREC_INT8=0,
		ePREC_INT16,
		ePREC_INT32,
		ePREC_FLOAT,
		ePREC_NONE
	};

	enum E_RESIZE_MODE
	{	// scaling modes
		eRESIZE_NEAREST=0,
		eRESIZE_BILINEAR,
		eRESIZE_BICUBIC,
		eNumResizeModes
	};
	const E_RESIZE_MODE keResizeModeDefault = eRESIZE_BICUBIC;

	enum E_PVRTC_ENCODING_MODE
	{
		ePVRTC_NORMAL=0,
		ePVRTC_ALT,
		ePVRTC_ALT2,
		eNumPVRTCModes
	};
	const E_PVRTC_ENCODING_MODE kePVRTCEncodingModeDefault = ePVRTC_NORMAL;

	const int kiPVRTCIterationsDefault = 8;

	enum E_ETC_ENCODING_MODE
	{
		eFast=0,
		eMedium,
		eSlow,
		eFastPerceptual,
		eMediumPerceptual,
		eSlowPerceptual,
		eNumETCModes
	};
	const E_ETC_ENCODING_MODE keETCEncodingModeDefault = eFastPerceptual;

	const char kstrETCEncodingMode[][18] =
	{
		"Fast",
		"Medium",
		"Slow",
		"Fast Perceptual",
		"Medium Perceptual",
		"Slow Perceptual",
	};

	enum E_BORDER_TYPE
	{
		eBORDER_PVRTC4=0,
		eBORDER_PVRTC2
	};
	const E_BORDER_TYPE keBorderTypeDefault = eBORDER_PVRTC4;
	
#ifdef __APPLE__
#pragma GCC visibility pop
#endif

}

#endif


/*****************************************************************************
End of file (PVRTexLibGlobals.h)
*****************************************************************************/
