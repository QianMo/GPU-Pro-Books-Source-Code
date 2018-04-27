#ifndef D3D10DRV_D3D10DEVICEQUERY_H_INCLUDED
#define D3D10DRV_D3D10DEVICEQUERY_H_INCLUDED

#include "Wrap3D/Src/DeviceQuery.h"

namespace Mod
{
	class D3D10DeviceQuery : public DeviceQuery
	{
	public:
		typedef DeviceQuery Base;
		typedef ComPtr< ID3D10Query > ResourcePtr;

		// construction/ destruction
	public:
		explicit D3D10DeviceQuery( const DeviceQueryConfig& cfg, ID3D10Device* dev );
		virtual ~D3D10DeviceQuery();

		// polymorphism
	private:
		virtual void BeginImpl()						OVERRIDE;
		virtual void EndImpl()							OVERRIDE;
		virtual bool GetDataImpl( void * data )	const	OVERRIDE;

		// data
	private:
		ResourcePtr mResource;

	};

}

#endif