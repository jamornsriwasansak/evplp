#pragma once

#include "GL/glew.h"
#include "glm/glm.hpp"

#include <iostream>
#include <vector>

class OpenglBuffer
{
public:
	static shared_ptr<OpenglBuffer> Create()
	{
		return make_shared<OpenglBuffer>();
	}

	OpenglBuffer()
	{
		glCreateBuffers(1, &mHandle);
	}

	void bufferVec3(const std::vector<glm::vec3> & meshData)
	{
		throw std::exception(); // "check your implementation"
		glNamedBufferData(mHandle, meshData.size() * sizeof(glm::vec3), meshData.data(), GL_STATIC_DRAW);
	}

	~OpenglBuffer()
	{
		glDeleteBuffers(1, &mHandle);
	}
	
	GLuint mHandle = 0;
};