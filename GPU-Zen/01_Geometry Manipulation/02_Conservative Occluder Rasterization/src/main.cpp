#include "Scenario.h"

#undef main
int main(int argc, const char** argv)
{
    CScenario scenario;
    scenario.Initialize( 1366, 768, "bengt" );
    scenario.Run();
    scenario.Terminate();

	return 0;
}