#ifndef COUTMETHODS_H
#define COUTMETHODS_H

#include "OpenGL.h"
#include "glm/glm.hpp"
#include "Scene/SceneDataStructs.h"
#include "Scene/ObjModel.h"

#include <iostream>

using namespace std;

void coutMatrix(const GLfloat* matrix);
void coutMatrix(const glm::mat4 matrix);
void coutVec(const glm::vec2& vec);
void coutVec(const glm::vec3& vec);
void coutVec(const glm::vec4& vec);

void coutSceneData(const SceneData& data);
void coutCommonElementData(const CommonElementData& elem);


#endif
