#pragma once

#include "GL/glew.h"
#include "glm/glm.hpp"

class OpenglQuery
{
public:
	
	OpenglQuery()
	{
		glGenQueries(1, &mHandle);
	}

	bool isAvailable()
	{
		GLint result = GL_FALSE;
		glGetQueryObjectiv(mHandle, GL_QUERY_RESULT_AVAILABLE, &result);
		return result == GL_TRUE;
	}

	void queryTimeStamp()
	{
		glQueryCounter(mHandle, GL_TIMESTAMP); 
	}

	void syncWait()
	{
		while (!isAvailable());
	}

	GLint64 syncGetQueryResult()
	{
		GLint64 result;
		syncWait();
		glGetQueryObjecti64v(mHandle, GL_QUERY_RESULT, &result);
		return result;
	}

	~OpenglQuery()
	{
		glDeleteQueries(1, &mHandle);
	}

	GLuint mHandle;
};
