#include "Precompiled.h"

#include "Math/Src/Operations.h"

#include "FormatHelpers.h"

namespace Mod
{

	using namespace Math;

	template<typename T1, typename T2>
	void ConvertFunc( T2 src, void* dest )
	{
		*(T1*)dest = (T1)src;
	}

	//------------------------------------------------------------------------

	template<typename T1>
	void ConvertFunc_UnpackUNORM( float src, void* dest )
	{
		*(T1*)dest = T1( std::max( std::min(src, 1.f ), 0.f ) * std::numeric_limits< T1 > :: max() );
	}

	//------------------------------------------------------------------------

	template<typename T1>
	void ConvertFunc_UnpackSNORM( float src, void* dest )
	{
		// follow DX9 documented signed packing ( i.e. -1.f transforms into -32767 not -32768 in SHORT4N )
		*(T1*)dest = T1( std::max( std::min(src, 1.f ), -1.f ) * std::numeric_limits< T1 > :: max() );
	}

	//------------------------------------------------------------------------

	template void ConvertFunc<float,	float>	(float,void*);
	
	template void ConvertFunc<INT32,	float>	(float,void*);
	template void ConvertFunc<INT16,	float>	(float,void*);
	template void ConvertFunc<INT8,		float>	(float,void*);

	template void ConvertFunc<UINT32,	float>	(float,void*);
	template void ConvertFunc<UINT16,	float>	(float,void*);
	template void ConvertFunc<UINT8,	float>	(float,void*);
	
	template void ConvertFunc<float,	int>	(int,void*);

	template void ConvertFunc<INT32,	int>	(int,void*);
	template void ConvertFunc<INT16,	int>	(int,void*);
	template void ConvertFunc<INT8,		int>	(int,void*);

	template void ConvertFunc<UINT32,	int>	(int,void*);
	template void ConvertFunc<UINT16,	int>	(int,void*);
	template void ConvertFunc<UINT8,	int>	(int,void*);

	//------------------------------------------------------------------------

	template void ConvertFunc_UnpackUNORM<UINT8>( float src, void* dest );
	template void ConvertFunc_UnpackUNORM<UINT16>( float src, void* dest );
	template void ConvertFunc_UnpackUNORM<UINT32>( float src, void* dest );

	template void ConvertFunc_UnpackSNORM<INT8>( float src, void* dest );
	template void ConvertFunc_UnpackSNORM<INT16>( float src, void* dest );
	template void ConvertFunc_UnpackSNORM<INT32>( float src, void* dest );

	//------------------------------------------------------------------------

	template<>
	void ConvertFunc<half,float>( float src, void* dest )
	{
		*(half*)dest = ftoh(src);
	}

	//------------------------------------------------------------------------	

	template<>
	void ConvertFunc<half,int>( int src, void* dest )
	{
		*(half*)dest = ftoh(float(src));
	}

}