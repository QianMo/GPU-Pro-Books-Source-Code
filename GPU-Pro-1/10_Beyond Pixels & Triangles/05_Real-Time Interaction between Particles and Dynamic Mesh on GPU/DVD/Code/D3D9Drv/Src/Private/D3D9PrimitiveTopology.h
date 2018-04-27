#ifndef D3D9DRV_D3D9PRIMITIVETOPOLOGY_H_INCLUDED
#define D3D9DRV_D3D9PRIMITIVETOPOLOGY_H_INCLUDED

#include "Wrap3D/Src/PrimitiveTopology.h"

namespace Mod
{
	class D3D9PrimitiveTopology : public PrimitiveTopology
	{
	public:
		explicit D3D9PrimitiveTopology( D3DPRIMITIVETYPE value );
		~D3D9PrimitiveTopology();

	public:
		D3DPRIMITIVETYPE GetD3D9Value() const;

		// data
	private:
		D3DPRIMITIVETYPE mValue;
	};
}

#endif