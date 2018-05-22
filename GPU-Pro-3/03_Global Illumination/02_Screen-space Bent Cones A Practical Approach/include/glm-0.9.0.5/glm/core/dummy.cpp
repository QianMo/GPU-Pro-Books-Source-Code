#include "../glm.hpp"
#include "../ext.hpp"

int main()
{
	//glm::mat2x3 m1(1.0f);
	//glm::vec3 v1(1.0f);
	//glm::vec2 w1 = m1 * v1;

	{
		glm::mat2x3 m(1.0f);
		glm::vec2 u(1.0f);
		glm::vec3 v(1.0f);
		float x = 1.0f;
		glm::vec3 a = m * u;
		glm::vec2 b = v * m;
		glm::mat2x3 n = x / m;
		glm::mat2x3 o = m / x;
		glm::mat2x3 p = x * m;
		glm::mat2x3 q = m * x;
	}

	{
		glm::mat2x4 m(1.0f);
		glm::vec2 u(1.0f);
		glm::vec4 v(1.0f);
		float x = 1.0f;
		glm::vec4 a = m * u;
		glm::vec2 b = v * m;
		glm::mat2x4 n = x / m;
		glm::mat2x4 o = m / x;
		glm::mat2x4 p = x * m;
		glm::mat2x4 q = m * x;
	}

	{
		glm::mat3x2 m(1.0f);
		glm::vec3 u(1.0f);
		glm::vec2 v(1.0f);
		float x = 1.0f;
		glm::vec2 a = m * u;
		glm::vec3 b = v * m;
		glm::mat3x2 n = x / m;
		glm::mat3x2 o = m / x;
		glm::mat3x2 p = x * m;
		glm::mat3x2 q = m * x;
	}

	{
		glm::mat3x4 m(1.0f);
		glm::vec3 u(1.0f);
		glm::vec4 v(1.0f);
		float x = 1.0f;
		glm::vec4 a = m * u;
		glm::vec3 b = v * m;
		glm::mat3x4 n = x / m;
		glm::mat3x4 o = m / x;
		glm::mat3x4 p = x * m;
		glm::mat3x4 q = m * x;
	}

	{
		glm::mat4x2 m(1.0f);
		glm::vec4 u(1.0f);
		glm::vec2 v(1.0f);
		float x = 1.0f;
		glm::vec2 a = m * u;
		glm::vec4 b = v * m;
		glm::mat4x2 n = x / m;
		glm::mat4x2 o = m / x;
		glm::mat4x2 p = x * m;
		glm::mat4x2 q = m * x;
	}

	{
		glm::mat4x3 m(1.0f);
		glm::vec4 u(1.0f);
		glm::vec3 v(1.0f);
		float x = 1.0f;
		glm::vec3 a = m * u;
		glm::vec4 b = v * m;
		glm::mat4x3 n = x / m;
		glm::mat4x3 o = m / x;
		glm::mat4x3 p = x * m;
		glm::mat4x3 q = m * x;
	}

	//{
	//	glm::mat3x4 m(1.0f);
	//	glm::vec3 v(1.0f);
	//	glm::vec4 w = m * v;
	//}
}
