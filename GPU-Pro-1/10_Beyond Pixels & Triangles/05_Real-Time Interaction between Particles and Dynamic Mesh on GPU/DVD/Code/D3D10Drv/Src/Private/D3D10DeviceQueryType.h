#ifndef D3D10DRV_D3D10DEVICEQUERYTYPE_H_INCLUDED
#define D3D10DRV_D3D10DEVICEQUERYTYPE_H_INCLUDED

#include "Wrap3D\Src\DeviceQueryType.h"
#include "Wrap3D\Src\DeviceQuery.h"

namespace Mod
{

	class D3D10DeviceQueryType : public DeviceQueryType
	{
		// types
	public:
		typedef DeviceQueryType Base;

		// construction/ destruction
	public:
		D3D10DeviceQueryType( D3D10_QUERY q, const type_info& dataTI, UINT64 size );
		virtual ~D3D10DeviceQueryType();

		// manipulation/ access
	public:
		D3D10_QUERY	GetValue() const;
		void		RemapData( void* dest, const void* src ) const;
		UINT64		GetDeviceDataSize() const;

		// polymorphism
	private:
		virtual const type_info& GetTypeInfoImpl() const OVERRIDE;

		// branch new polymorphism
	private:
		virtual void RemapDataImpl( void* dest, const void* src ) const = 0;

	private:
		D3D10_QUERY			mValue;
		const type_info&	mDataTypeInfo;
		UINT64				mDeviceDataSize;

	};

	//------------------------------------------------------------------------

	template <D3D10_QUERY q>
	class D3D10DeviceQueryTypeImpl :	public D3D10DeviceQueryType
	{
		// types
	public:
		typedef D3D10DeviceQueryType Base;
		static const D3D10_QUERY Query = q;

		// construction/ destruction
	public:
		D3D10DeviceQueryTypeImpl();		
		virtual ~D3D10DeviceQueryTypeImpl();

		// polymorphism
	private:
		virtual void RemapDataImpl( void* dest, const void* src ) const OVERRIDE;

	};

	//------------------------------------------------------------------------

}


#endif