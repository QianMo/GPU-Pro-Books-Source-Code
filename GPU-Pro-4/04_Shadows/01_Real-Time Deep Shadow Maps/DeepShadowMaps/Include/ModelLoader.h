#pragma once
#include "Model.h"

CoreResult LoadHairModel(Core *core, std::wstring &filename, Model *model);
CoreResult LoadXModel(Core *core, std::wstring &filename, Model ***models, int *numModels);