#pragma once

namespace Mesh {

	/// Identifier for different ways meshes can be rendered. 
	class Role
	{
		unsigned int id;
		static unsigned int nextId;
		
	public:

		Role();
		bool operator<(const Role& co) const;
		bool operator==(const Role& co) const;

		static const Role invalid;

		inline bool isValid() const {return id != 0;}

		struct Compare
		{
			bool operator() (const Role& a, const Role& b) const
			{
				return a < b;
			}
		};
	};

}