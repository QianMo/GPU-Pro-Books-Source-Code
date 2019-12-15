#/usr/bin/perl
use strict;

#configuration
my $max_param_count = 10;

#implementation
open(OUTPUT, ">delegate_list.hpp") || die $!;
print(OUTPUT "// Automaticly generaged by $0\n\n");

my @param_list;
for (my $param_count = 0; $param_count <= $max_param_count;
		push(@param_list, ++$param_count))
{
	print(OUTPUT "// $param_count params\n");

	print(OUTPUT "#define SRUTIL_DELEGATE_PARAM_COUNT $param_count\n");
	
	print(OUTPUT "#define SRUTIL_DELEGATE_TEMPLATE_PARAMS ",
		join(", ", map({"typename A$_"} @param_list)), "\n");

	print(OUTPUT "#define SRUTIL_DELEGATE_TEMPLATE_ARGS ",
		join(", ", map({"A$_"} @param_list)), "\n");

	print(OUTPUT "#define SRUTIL_DELEGATE_PARAMS ",
		join(", ", map({"A$_ a$_"} @param_list)), "\n");

	print(OUTPUT "#define SRUTIL_DELEGATE_ARGS ",
		join(",", map({"a$_"} @param_list)), "\n");

	print(OUTPUT "#define SRUTIL_DELEGATE_INVOKER_INITIALIZATION_LIST ",
		join(",", map({"a$_(a$_)"} @param_list)), "\n");

	print(OUTPUT "#define SRUTIL_DELEGATE_INVOKER_DATA ",
		map({"A$_ a$_;"} @param_list), "\n");

	print(OUTPUT '#include "delegate_template.hpp"', "\n",
		"#undef SRUTIL_DELEGATE_PARAM_COUNT\n",
		"#undef SRUTIL_DELEGATE_TEMPLATE_PARAMS\n",
		"#undef SRUTIL_DELEGATE_TEMPLATE_ARGS\n",
		"#undef SRUTIL_DELEGATE_PARAMS\n",
		"#undef SRUTIL_DELEGATE_ARGS\n",
		"#undef SRUTIL_DELEGATE_INVOKER_INITIALIZATION_LIST\n",
		"#undef SRUTIL_DELEGATE_INVOKER_DATA\n",
		"\n");
}
close(OUTPUT) || die $!;
0;
