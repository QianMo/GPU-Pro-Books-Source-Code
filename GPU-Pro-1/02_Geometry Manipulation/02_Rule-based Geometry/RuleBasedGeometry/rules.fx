#include "symbolids.fx"
#include "moduleoperations.fx"

//*************************************************
// Rule shaders
//	- output is the nth module of the rule successor
//	- output can depend on the predecessor (parent
//		module)
//*************************************************

// Rule: SYMBOL_S1 -> SYMBOL_S2
Module getNextModule_SYMBOL_S1_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S2;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S2) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S10 -> SYMBOL_S11
Module getNextModule_SYMBOL_S10_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S11;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S11) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S11 -> SYMBOL_S12
Module getNextModule_SYMBOL_S11_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S12;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S12) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S12 -> SYMBOL_S13
Module getNextModule_SYMBOL_S12_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S13;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S13) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S13 -> SYMBOL_S14
Module getNextModule_SYMBOL_S13_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S14;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S14) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S14 -> SYMBOL_S15
Module getNextModule_SYMBOL_S14_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S15;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S15) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S15 -> SYMBOL_S16
Module getNextModule_SYMBOL_S15_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S16;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S16) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S16 -> SYMBOL_S17
Module getNextModule_SYMBOL_S16_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S17;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S17) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S17 -> SYMBOL_S18
Module getNextModule_SYMBOL_S17_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S18;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S18) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S18 -> SYMBOL_S19
Module getNextModule_SYMBOL_S18_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S19;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S19) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S19 -> SYMBOL_S20
Module getNextModule_SYMBOL_S19_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S20;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S20) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S2 -> SYMBOL_S3
Module getNextModule_SYMBOL_S2_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S3;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S3) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S20 -> SYMBOL_S21
Module getNextModule_SYMBOL_S20_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S21;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S21) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S21 -> SYMBOL_S22
Module getNextModule_SYMBOL_S21_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S22;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S22) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S22 -> SYMBOL_S23
Module getNextModule_SYMBOL_S22_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S23;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S23) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S23 -> SYMBOL_S24
Module getNextModule_SYMBOL_S23_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S24;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S24) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S24 -> SYMBOL_S25
Module getNextModule_SYMBOL_S24_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S25;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S25) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S25 -> SYMBOL_S26
Module getNextModule_SYMBOL_S25_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S26;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S26) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S26 -> SYMBOL_S27
Module getNextModule_SYMBOL_S26_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S27;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S27) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S27 -> SYMBOL_S28
Module getNextModule_SYMBOL_S27_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S28;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S28) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S28 -> SYMBOL_S29
Module getNextModule_SYMBOL_S28_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S29;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S29) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S29 -> SYMBOL_S30
Module getNextModule_SYMBOL_S29_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S30;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S30) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S3 -> SYMBOL_S4
Module getNextModule_SYMBOL_S3_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S4;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S4) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S30 -> SYMBOL_S31
Module getNextModule_SYMBOL_S30_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S31;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S31) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S31 -> SYMBOL_S32
Module getNextModule_SYMBOL_S31_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S32;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S32) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S32 -> SYMBOL_S33
Module getNextModule_SYMBOL_S32_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S33;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S33) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S33 -> SYMBOL_S34
Module getNextModule_SYMBOL_S33_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S34;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S34) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S34 -> SYMBOL_S35
Module getNextModule_SYMBOL_S34_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S35;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S35) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S35 -> SYMBOL_S36
Module getNextModule_SYMBOL_S35_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S36;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S36) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S36 -> SYMBOL_S37
Module getNextModule_SYMBOL_S36_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S37;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S37) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S37 -> SYMBOL_S38
Module getNextModule_SYMBOL_S37_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S38;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S38) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S38 -> SYMBOL_S39
Module getNextModule_SYMBOL_S38_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S39;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S39) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S39 -> SYMBOL_S40
Module getNextModule_SYMBOL_S39_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S40;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S40) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S4 -> SYMBOL_S5
Module getNextModule_SYMBOL_S4_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S5;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S5) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S40 -> SYMBOL_S41
Module getNextModule_SYMBOL_S40_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S41;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S41) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S41 -> SYMBOL_S42
Module getNextModule_SYMBOL_S41_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S42;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S42) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S42 -> SYMBOL_S43
Module getNextModule_SYMBOL_S42_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S43;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S43) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S43 -> SYMBOL_S44
Module getNextModule_SYMBOL_S43_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S44;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S44) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S44 -> SYMBOL_S45
Module getNextModule_SYMBOL_S44_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S45;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S45) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S45 -> SYMBOL_S46
Module getNextModule_SYMBOL_S45_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S46;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S46) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S46 -> SYMBOL_S47
Module getNextModule_SYMBOL_S46_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S47;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S47) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S47 -> SYMBOL_S48
Module getNextModule_SYMBOL_S47_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S48;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S48) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S48 -> SYMBOL_S49
Module getNextModule_SYMBOL_S48_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S49;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S49) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S49 -> SYMBOL_S50
Module getNextModule_SYMBOL_S49_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S50;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S50) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S5 -> SYMBOL_S6
Module getNextModule_SYMBOL_S5_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S6;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S6) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S50 -> SYMBOL_S1
Module getNextModule_SYMBOL_S50_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S1;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S1) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S6 -> SYMBOL_S7
Module getNextModule_SYMBOL_S6_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S7;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S7) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S7 -> SYMBOL_S8
Module getNextModule_SYMBOL_S7_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S8;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S8) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S8 -> SYMBOL_S9
Module getNextModule_SYMBOL_S8_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S9;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S9) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

// Rule: SYMBOL_S9 -> SYMBOL_S10
Module getNextModule_SYMBOL_S9_RULE_1( Module parent, int number )
{
	Module output = parent;
	float dummyVariable;

	// TODO: Implement module initialization (i.e. rule behavior) here
	//	- set symbol ID (done already, modify if needed)
	//	- set module parameters (e.g. position)

	switch( number )
	{
		case 1:
			output.symbolID = SYMBOL_S10;

			// Module operations generated automatically from the grammar file:
			module_resize( output, parent, 0.5 );

			// TODO: implement initialization of 1. module in the successor (SYMBOL_S10) here

			break;
		default:
			output.symbolID = SYMBOL_INVALID;
			break;
	};

	return output;
}

