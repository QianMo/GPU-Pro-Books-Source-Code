#ifndef D3D9DRV_FORMATHELEPRS_H_INCLUDED
#define D3D9DRV_FORMATHELEPRS_H_INCLUDED

#include "Wrap3D\Src\Format.h"
#include "Math\Src\Types.h"

#include "D3D9ExtraFormats.h"

namespace Mod
{
	template<typename T1, typename T2>
	void ConvertFunc( T2 src, void* dest );

	template<typename T1>
	void ConvertFunc_UnpackUNORM( float src, void* dest );

	template<typename T1>
	void ConvertFunc_UnpackSNORM( float src, void* dest );

	template <D3DFORMAT F>
	struct FormatTraits;

	template < typename T, UINT32 CC>
	struct DeriveFmtSpecificsHelper
	{
		typedef T Result;
	};

	template <typename T, typename U>
	struct DeriveFmtSpecifics
	{
		static const UINT32 ComponentCount = 1;
		typedef T CompType;
	};

	template <typename T>
	struct DeriveFmtSpecifics<T, typename DeriveFmtSpecificsHelper<T, T::COMPONENT_COUNT>::Result>
	{
		static const UINT32 ComponentCount = T::COMPONENT_COUNT;
		typedef typename T::comp_type CompType;
	};

#define DECLARE_D3DFORMAT_STRUCT(format, bit_count, type)											\
			template<>																				\
			struct FormatTraits<D3DFORMAT(format)>													\
			{																						\
				typedef type Type;																	\
				typedef DeriveFmtSpecifics<Type,Type>::CompType CompType;							\
				enum																				\
				{																					\
					BitCount = bit_count,															\
					Format = D3DFORMAT(format),														\
					ComponentCount = DeriveFmtSpecifics<Type,Type>::ComponentCount					\
				};																					\
			};

DECLARE_D3DFORMAT_STRUCT( D3DFMT_R8G8B8				, 3*8	, Math::ubyte3	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A8R8G8B8			, 4*8	, Math::ubyte4	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_X8R8G8B8			, 4*8	, Math::ubyte4	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_R5G6B5				, 16	, Math::ushort	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_X1R5G5B5			, 16	, Math::ushort	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A1R5G5B5			, 16	, Math::ushort	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A4R4G4B4			, 16	, Math::ushort	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_R3G3B2				, 8		, Math::ubyte	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A8					, 8		, Math::ubyte	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A8R3G3B2			, 16	, Math::ushort	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_X4R4G4B4			, 16	, Math::ushort	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A2B10G10R10		, 32	, Math::uint	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A8B8G8R8			, 4*8	, Math::ubyte4	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_X8B8G8R8			, 4*8	, Math::ubyte4	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_G16R16				, 2*16	, Math::short2	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A2R10G10B10		, 32	, Math::uint	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A16B16G16R16		, 4*16	, Math::short4	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A8P8				, 2*8	, Math::ubyte2	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_P8					, 8		, Math::ubyte	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_L8					, 8		, Math::ubyte	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A8L8				, 2*8	, Math::ubyte2	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A4L4				, 2*4	, Math::ubyte	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_V8U8				, 2*8	, Math::ubyte2	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_L6V5U5				, 16 	, Math::ushort	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_X8L8V8U8			, 4*8	, Math::ubyte4	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_Q8W8V8U8			, 4*8	, Math::ubyte4	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_V16U16				, 2*16	, Math::short2	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A2W10V10U10		, 32	, Math::uint	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_UYVY				, 0		, Math::uint	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_R8G8_B8G8			, 4*8	, Math::ubyte4	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_YUY2				, 0		, Math::uint	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_G8R8_G8B8			, 4*8	, Math::ubyte4	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_DXT1				, 4		, Math::ubyte	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_DXT2				, 8		, Math::ubyte	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_DXT3				, 8		, Math::ubyte	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_DXT4				, 8		, Math::ubyte	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_DXT5				, 8		, Math::ubyte	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_D16_LOCKABLE		, 16	, Math::ushort	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_D32				, 32	, Math::uint	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_D15S1				, 16	, Math::ushort	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_D24S8				, 32	, Math::uint	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_D24X8				, 32	, Math::uint	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_D24X4S4			, 32	, Math::uint	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_D16				, 16	, Math::ushort	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_D32F_LOCKABLE		, 32	, Math::uint	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_D24FS8				, 32	, Math::uint	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_L16				, 16	, Math::ushort	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_VERTEXDATA			, 0		, Math::ubyte	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_INDEX16			, 16	, Math::ushort	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_INDEX32			, 32	, Math::uint	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_Q16W16V16U16		, 4*16	, Math::ushort4	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_MULTI2_ARGB8		, 4*8	, Math::ubyte2	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_R16F				, 16	, Math::half	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_G16R16F			, 2*16	, Math::half2	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A16B16G16R16F		, 4*16	, Math::half4	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_R32F				, 32	, float			)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_G32R32F			, 2*32	, Math::float2	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_A32B32G32R32F		, 4*32	, Math::float4	)
DECLARE_D3DFORMAT_STRUCT( D3DFMT_CxV8U8				, 2*8	, Math::ubyte2	)

// our formats
DECLARE_D3DFORMAT_STRUCT( MDFMT_R32G32B32_FLOAT		, 3*32	, Math::float3	)

}

#endif