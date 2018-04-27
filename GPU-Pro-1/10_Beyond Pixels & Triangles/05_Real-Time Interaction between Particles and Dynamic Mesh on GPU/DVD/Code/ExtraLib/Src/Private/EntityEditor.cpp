#include "Precompiled.h"

#include "Common/Src/VarType.h"

#include "WrapSys/Src/System.h"

#include "Providers/Src/TextureProvider.h"
#include "Providers/Src/Providers.h"

#include "SceneRender/Src/EffectParamSet.h"
#include "SceneRender/Src/EntityCmptConfig.h"
#include "SceneRender/Src/EntityCmpt.h"
#include "SceneRender/Src/Entity.h"

#include "EntityEditorConfig.h"
#include "EntityEditor.h"

#define MD_NAMESPACE EntityEditorNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	EntityEditor::EntityEditor( const EntityEditorConfig& cfg ) :
	Parent( cfg )
	{
	}

	//------------------------------------------------------------------------

	EntityEditor::~EntityEditor() 
	{
	}

	//------------------------------------------------------------------------

	void
	EntityEditor::UpdateParam( 	const TargetCmpts& targetCmpts,
								const AnsiString& name,								
								const String& value 
								) const
	{
		const Entity::EntityCmpts& cmpts = GetConfig().ent->GetEntityCmpts();

		for( size_t i = 0, e = targetCmpts.size(); i < e; i ++ )
		{
			UINT32 idx = targetCmpts[ i ];
			MD_FERROR_ON_FALSE( idx < cmpts.size() );

			const EffectParamSetPtr& eps = cmpts[ idx ]->GetConfig().mEffectParamSet;

			if( eps->GetParamType( name ) == VarType::SHADER_RESOURCE )
			{
				String fullPath = GetFullPath( Providers::Single().GetTextureProv(), value );
				if( !System::Single().FileExists( fullPath ) )
					continue;				
			}

			eps->UpdateEffectParam( name, value );
		}
	}

}