#ifndef EXTRALIB_ENTITYEDITOR_H_INCLUDED
#define EXTRALIB_ENTITYEDITOR_H_INCLUDED

#include "Forw.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE EntityEditorNS
#include "ConfigurableImpl.h"

namespace Mod
{

	// NOTE : TO BE USED IN EDITORS AND NOT THE GAME ENVIRONMENT ;)
	class EntityEditor : public EntityEditorNS::ConfigurableImpl<EntityEditorConfig>
	{
		// types
	public:
		typedef Types< UINT32 > :: Vec TargetCmpts;

		// constructors / destructors
	public:
		explicit EntityEditor( const EntityEditorConfig& cfg );
		~EntityEditor();
	
		// manipulation/ access
	public:
		void UpdateParam(	const TargetCmpts& targetCmpts,
							const AnsiString& name,							
							const String& value 
							) const;


		// data
	private:

	};
}

#endif