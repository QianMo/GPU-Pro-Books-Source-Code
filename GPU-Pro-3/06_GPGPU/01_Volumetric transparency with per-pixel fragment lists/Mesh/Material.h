#pragma once
#include "d3dx11effect.h"

namespace Mesh
{
	/// Material for rendering a mesh. Contains the effect framework settings.
	class Material
	{
		ID3DX11EffectPass* pass;
		unsigned int flags;

		typedef std::map<ID3DX11EffectVariable*, void*>		EffectVariableSettings;
		EffectVariableSettings effectVariableSettings;

		Material(ID3DX11EffectPass* pass, unsigned int flags):pass(pass),flags(flags)
		{
		}
	public:
		/// Shared pointer type.
		typedef boost::shared_ptr<Material> P;
		/// Invokes contructor, returns shared pointer.
		static Material::P make(ID3DX11EffectPass* pass, unsigned int flags) { return Material::P(new Material(pass, flags)); }

		~Material();

		/// Saves a current effect setting to the material. (Use effect API to access variable.)
		void saveVariable(ID3DX11EffectVariable* variable);
		/// Applies saved variables to effect state.
		void apply(ID3D11DeviceContext* context);

	};

} // namespace Mesh
