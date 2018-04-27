#ifndef PROVIDERS_MODEL_H_INCLUDED
#define PROVIDERS_MODEL_H_INCLUDED

#include "Math/Src/BBox.h"
#include "Wrap3D/Src/Forw.h"

#include "Forw.h"

#include "ModelConfig.h"

#include "ExportDefs.h"

#define MD_NAMESPACE ModelNS
#include "ConfigurableImpl.h"

namespace Mod
{
	class Model : public ModelNS::ConfigurableImpl< ModelConfig >
	{
		// types
	public:
		typedef ModelConfig::CmptEntries CmptEntries;

		// construction/ destruction
	public:
		EXP_IMP explicit Model( const ModelConfig& cfg );
		EXP_IMP virtual ~Model();

		// manipulation/ access
	public:
		EXP_IMP const Math::BBox&	GetBBox() const;
		EXP_IMP const CmptEntries&	GetCmpts() const;

		// data
	private:
		Math::BBox	mBBox;		

	};
}

#endif