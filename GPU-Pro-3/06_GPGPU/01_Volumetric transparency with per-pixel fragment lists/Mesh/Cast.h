#pragma once
#include "Mesh/Shaded.h"
#include "Mesh/Role.h"

namespace Mesh
{
	/// Mesh that can be shaded in multiple roles. A collection of shaded meshes.
	class Cast
	{
		typedef std::map<Role, Shaded::P> RoleShadedMap;
		RoleShadedMap roleShadedMap;

		Cast();
	public:
		/// Shared pointer type.
		typedef boost::shared_ptr<Cast> P;
		/// Invokes constructor, wraps pointer into shared pointer.
		static Cast::P make() { return Cast::P(new Cast());}

		~Cast(void);

		/// Adds a shaded mesh.
		void add(Role role, Shaded::P shaded);

		/// Draws shaded mesh indicated by role.
		void draw(ID3D11DeviceContext* context, Role role);
	};

} // namespace Mesh
