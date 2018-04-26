#pragma once

class Role
{
	friend class Play;

	unsigned int id;
	
	static unsigned int nextId;
	Role();
public:
	Role(const Role& o);
	bool operator<(const Role& o) const;
	bool operator==(const Role& o) const;

	static const Role invalid;

	inline bool isValid() const {return id != 0;}
};
