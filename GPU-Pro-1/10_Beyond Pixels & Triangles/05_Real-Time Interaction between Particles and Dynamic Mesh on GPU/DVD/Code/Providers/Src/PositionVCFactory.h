#ifndef PROVIDERS_POSITIONVCFACTORY_H_INCLUDED
#define PROVIDERS_POSITIONVCFACTORY_H_INCLUDED

#include "Wrap3D/Src/Forw.h"

#include "VertexCmptFactory.h"

namespace Mod
{
	class PositionVCFactory :	public VertexCmptFactory
	{
		// types
	public:
		typedef VertexCmptFactory Base;

		// construction/ destruction
	public:
		explicit PositionVCFactory( const VertexCmptFactoryConfig& cfg );
		virtual ~PositionVCFactory();

		// polymorphism
	private:
		virtual void ModifyILElemConfigImpl( ILElement& el )  const OVERRIDE;
		virtual void ConstructVertexDataImpl( const RawVertexData& rawData, UINT32 idx, UINT32 disp, const Format* fmt, Bytes& oData ) const OVERRIDE;

	};
}

#endif