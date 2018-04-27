#ifndef D3D10DRV_D3D10PRIMITIVETOPOLOGY_H_INCLUDED
#define D3D10DRV_D3D10PRIMITIVETOPOLOGY_H_INCLUDED

#include "Wrap3D/Src/PrimitiveTopology.h"

namespace Mod
{
	class D3D10PrimitiveTopology : public PrimitiveTopology
	{
	public:
		explicit D3D10PrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY value );
		~D3D10PrimitiveTopology();

		D3D10_PRIMITIVE_TOPOLOGY GetD3D10Value() const;

		// data
	private:
		D3D10_PRIMITIVE_TOPOLOGY mValue;
	};
}

#endif