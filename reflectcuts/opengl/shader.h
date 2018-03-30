#pragma once

#include "common/reflectcuts.h"
#include "GL/glew.h"

#include <string>
#include <exception>
#include <fstream>
#include <vector>

class OpenglUniform
{
public:
	static std::shared_ptr<OpenglUniform> CreateFromHandle(const GLuint handle)
	{
		return make_shared<OpenglUniform>(handle);
	}

	OpenglUniform()
	{
		mHandle = -1;
	}

	OpenglUniform(const GLuint handle)
	{
		mHandle = handle;
	}

	void setUniform(const bool v) const
	{
		#ifndef REALTIME_PERFORMANCE
		if (!mActive) { std::cout << mName << " is not active" << std::endl; return; }
		#endif
		glUniform1i(mHandle, v);
	}

	void setUniform(const int v) const
	{
		#ifndef REALTIME_PERFORMANCE
		if (!mActive) { std::cout << mName << " is not active" << std::endl; return; }
		#endif
		glUniform1i(mHandle, v);
	}

	void setUniform(const float v) const
	{
		#ifndef REALTIME_PERFORMANCE
		if (!mActive) { std::cout << mName << " is not active" << std::endl; return; }
		#endif
		glUniform1f(mHandle, v);
	}

	void setUniform(const double v) const
	{
		#ifndef REALTIME_PERFORMANCE
		if (!mActive) { std::cout << mName << " is not active" << std::endl; return; }
		#endif
		glUniform1f(mHandle, (float)v);
	}

	void setUniform(const glm::vec2 &v) const
	{
		#ifndef REALTIME_PERFORMANCE
		if (!mActive) { std::cout << mName << " is not active" << std::endl; return; }
		#endif
		glUniform2fv(mHandle, 1, &v[0]);
	}

	void setUniform(const glm::vec3 &v) const
	{
		#ifndef REALTIME_PERFORMANCE
		if (!mActive) { std::cout << mName << " is not active" << std::endl; return; }
		#endif
		glUniform3fv(mHandle, 1, &v[0]);
	}

	void setUniform(const glm::vec4 &v) const
	{
		#ifndef REALTIME_PERFORMANCE
		if (!mActive) { std::cout << mName << " is not active" << std::endl; return; }
		#endif
		glUniform4fv(mHandle, 1, &v[0]);
	}

	void setUniform(const glm::mat4 &v) const
	{
		#ifndef REALTIME_PERFORMANCE
		if (!mActive) { std::cout << mName << " is not active" << std::endl; return; }
		#endif
		glUniformMatrix4fv(mHandle, 1, GL_FALSE, &v[0][0]);
	}

	void setUniformArray(const glm::mat4 *v, const GLsizei numMatrices) const
	{
		#ifdef REALTIME_PERFORMANCE
		if (!mActive) { std::cout << mName << " is not active" << std::endl; return; }
		#endif
		glUniformMatrix4fv(mHandle, numMatrices, GL_FALSE, &((*v)[0][0]));
	}

	bool mActive = false;
	GLint mHandle;
	std::string mName;
};

template<const GLenum EShaderType>
class OpenglShader
{
public:
	static std::shared_ptr<OpenglShader> CreateFromFile(const std::string & filepath)
	{
		shared_ptr<OpenglShader> result = make_shared<OpenglShader>();
		result->mLoadFromFile = true;
		result->mFilepath = filepath;
		result->recompile();
		return result;
	}

	OpenglShader()
	{
		mHandle = glCreateShader(EShaderType);
	}

	~OpenglShader()
	{
		glDeleteShader(mHandle);
	}

	inline void compile(const std::string & shaderString)
	{
		// get infolog
		const GLchar * shaderStringPtr = shaderString.c_str();
		glShaderSource(mHandle, 1, &shaderStringPtr, NULL);
		glCompileShader(mHandle);

		GLint shaderCompileResult = GL_FALSE;
		int shaderInfoLogLength;
		glGetShaderiv(mHandle, GL_COMPILE_STATUS, &shaderCompileResult);
		glGetShaderiv(mHandle, GL_INFO_LOG_LENGTH, &shaderInfoLogLength);

		// check if info exists?
		if (shaderInfoLogLength > 1)
		{
			std::string shaderInfoLogString(shaderInfoLogLength + 1, (char)0);
			glGetShaderInfoLog(mHandle, shaderInfoLogLength, NULL, (GLchar *)shaderInfoLogString.c_str());
			std::cout << "ShaderLog : " << shaderInfoLogString << " from " << mFilepath << std::endl;
		}
	}

	inline void recompile()
	{
		if (mLoadFromFile)
		{
			// load shader string
			std::ifstream ifs(mFilepath);
			if (!ifs.is_open()) {
				std::cerr << "Err: Impossible to Open " << mFilepath << "\n";
				assert(false);
			}
			std::string shaderString((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
			this->compile(shaderString);
		}
	}

	bool mLoadFromFile;
	std::string mFilepath;

	// the latest shader string that still compilable
	std::string mCompiledShaderString;
	GLuint mHandle;
};

using OpenglVertexShader = OpenglShader<GL_VERTEX_SHADER>;
using OpenglGeometryShader = OpenglShader<GL_GEOMETRY_SHADER>;
//using OpenglComputeShader = OpenglShader<GL_COMPUTE_SHADER>;
using OpenglFragmentShader = OpenglShader<GL_FRAGMENT_SHADER>;

class OpenglProgram
{
public:

	OpenglProgram()
	{
		mHandle = glCreateProgram();
	}

	void attachShader(shared_ptr<OpenglVertexShader> shader)
	{
		assert(shader != nullptr);
		mVertexShader = shader;
		glAttachShader(mHandle, shader->mHandle);
	}

	void attachShader(shared_ptr<OpenglFragmentShader> shader)
	{
		assert(shader != nullptr);
		mFragmentShader = shader;
		glAttachShader(mHandle, shader->mHandle);
	}

	void attachShader(shared_ptr<OpenglGeometryShader> shader)
	{
		assert(shader != nullptr);
		mGeometryShader = shader;
		glAttachShader(mHandle, shader->mHandle);
	}

	void compile()
	{
		glLinkProgram(mHandle);
		// get infolog
		GLint programCompileResult = GL_FALSE;
		int programInfoLogLength;
		glGetProgramiv(mHandle, GL_LINK_STATUS, &programCompileResult);
		glGetProgramiv(mHandle, GL_INFO_LOG_LENGTH, &programInfoLogLength);

		// check if info exists?
		if (programInfoLogLength > 1)
		{
			std::string programInfoLogString(programInfoLogLength + 1, char(0));
			glGetProgramInfoLog(mHandle, programInfoLogLength, NULL, (GLchar *)programInfoLogString.c_str());
			std::cout << "OpenGLProgram's Log : " << std::endl << programInfoLogString << std::endl;
		}
	}

	void recompile()
	{
		glDeleteProgram(mHandle);
		mHandle = glCreateProgram();

		// reload every shader
		if (mVertexShader != nullptr) { mVertexShader->recompile(); attachShader(mVertexShader); }
		if (mFragmentShader != nullptr) { mFragmentShader->recompile(); attachShader(mFragmentShader); }
		if (mGeometryShader != nullptr) { mGeometryShader->recompile(); attachShader(mGeometryShader); }

		compile();
	}

	void initializeUniform(shared_ptr<OpenglUniform> & uniform)
	{
		uniform->mHandle = glGetUniformLocation(this->mHandle, uniform->mName.c_str());
		if (uniform->mHandle == -1)
		{
			uniform->mActive = false;
			std::cout << "couldn't find uniform name : " << uniform->mName << std::endl;
		}
		else
		{
			uniform->mActive = true;
		}
	}

	shared_ptr<OpenglUniform> registerUniform(const std::string & uniformName)
	{
		shared_ptr<OpenglUniform> result = make_shared<OpenglUniform>();
		result->mName = uniformName;
		initializeUniform(result);
		return result;
	}

	~OpenglProgram()
	{
		glDeleteProgram(mHandle);
	}

	shared_ptr<OpenglVertexShader> mVertexShader = nullptr;
	shared_ptr<OpenglFragmentShader> mFragmentShader = nullptr;
	shared_ptr<OpenglGeometryShader> mGeometryShader = nullptr;

	std::vector<OpenglUniform> mUniforms;

	GLuint mHandle;
};