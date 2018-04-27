#ifndef COMMON_NAMED_H_INCLUDED
#define COMMON_NAMED_H_INCLUDED

namespace Mod
{
	template <typename S >
	class BasicNamed
	{
		// types
	public:
		typedef S StringType;

		// construction/ destruction
	public:
		explicit BasicNamed(const StringType& name);

	protected:
		~BasicNamed();

		// manipulation/ access
	public:
		const StringType&						GetName() const;
		const typename StringType::value_type*	GetCName() const; // does .c_str() call, so follow the rulz

		// derivee access
	protected:
		void Rename(const StringType& name);

		// data
	private:
		StringType mName;
	};

	typedef BasicNamed< String >		Named;
	typedef BasicNamed< AnsiString >	AnsiNamed;
}

#endif