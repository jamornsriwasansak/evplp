#pragma once

#include <iostream>
#include <exception>
#if defined(__GNUC__)
#else
#include <conio.h>
#endif
#include <sstream>
#include <cassert>

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#define REALTIME_PERFORMANCE

#if defined(__GNUC__)
#else
#if 1
// Force using dedicated Nvidia card
extern "C"
{
	__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

// Force using dedicated AMD card
extern "C"
{
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif
#endif

#if 1
#define glCheckError() { \
    GLenum err = glGetError(); \
    if (err != GL_NO_ERROR) { \
        fprintf(stderr, "glCheckError: %04x caught at %s:%u\n", err, __FILE__, __LINE__); \
        rt_assert(0); \
    } \
}
#else
#define glCheckError()
#endif

class RealTime
{
public:
	RealTime()
	{
	}

	void setup(const glm::uvec2 & resolution)
	{
		// setup glfw3
		if (!glfwInit())
		{
			throw std::exception(); // "GLFW init failed"
			return;
		}

		glfwWindowHint(GLFW_SAMPLES, 1);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OPENGL_VERSION_MAJOR);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OPENGL_VERSION_MINOR);
		//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
		//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		mWindow = glfwCreateWindow(resolution.x, resolution.y, "work please", NULL, NULL);
		assert(mWindow != nullptr);

		glfwMakeContextCurrent(mWindow);
		// turn off vsync
		glfwSwapInterval(0);

		// setup glew
		glewExperimental = true;
		GLenum error = glewInit();
		if (error != GLEW_OK)
		{
			std::cout << glewGetErrorString(error) << std::endl;
			throw std::exception(); // "GLEW init failed"
			return;
		}

		#ifndef REALTIME_PERFORMANCE
			// force using ARB
			assert(GLEW_ARB_debug_output);
			glEnable(GL_DEBUG_OUTPUT);
			glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
			glDebugMessageCallback(OpenglErrorCallback, nullptr);
			GLuint unusedIds = 0;
			glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, &unusedIds, true);
		#endif

		// force using direct state access
		assert(GLEW_EXT_direct_state_access);
		glCreateVertexArrays(1, &mGlobalVaoId);
		glBindVertexArray(mGlobalVaoId);
	}

	void loop(const std::function<bool(std::string * titleExtend)> & beforeSwap, const std::function<bool(std::string * titleExtend)> & afterSwap = nullptr)
	{
		StopWatch sw;
		sw.reset();
		bool result = false;
		double frameCount = 0;
		std::string titleExtend;
		do
		{
			titleExtend = "";

			result = beforeSwap(&titleExtend);
			if (result)
			{
				// Swap buffers
				glfwSwapBuffers(mWindow);
				glfwPollEvents();
			}

			if (afterSwap != nullptr)
			{
				result = result && afterSwap(&titleExtend);
			}

			frameCount++;

			long long timeSpent = sw.timeMilliSec();
			if (timeSpent >= 1000)
			{
				double dTimeSpent = (double)timeSpent;
				double fps = frameCount / dTimeSpent * 1000;
				double spf = dTimeSpent / frameCount;
				std::string title = std::to_string(fps) + "fps, " + std::to_string(spf) + "ms, " + titleExtend;
				this->setWindowTitle(title);

				// reset everything
				frameCount = 0;
				sw.reset();
			}
		} // Check if the ESC key was pressed or the window was closed
		while (result && glfwGetKey(mWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(mWindow) == 0);
	}

	void destroy()
	{
		glfwDestroyWindow(mWindow);
	}

	//--------------- window stuffs -----------------//

	void setWindowTitle(const std::string & title)
	{
		glfwSetWindowTitle(mWindow, title.c_str());
	}

	Vec2 cursorPos()
	{
		glm::dvec2 pos;
		glfwGetCursorPos(mWindow, &pos.x, &pos.y);
		return pos;
	}

	~RealTime()
	{
	}

	GLFWwindow * mWindow;
	GLuint mGlobalVaoId;

private:

	static void APIENTRY OpenglErrorCallback(GLenum source,
		GLenum type,
		GLuint id,
		GLenum severity,
		GLsizei length,
		const GLchar* message,
		const void* userParam)
	{
		std::stringstream ss;
		ss <<  "---------------------opengl-callback-start------------" << std::endl;
		ss << "message: " << message << std::endl;
		ss << "type: ";

		bool isError = false;

		switch (type)
		{
		case GL_DEBUG_TYPE_ERROR:
			ss << "ERROR";
			isError = true;
			break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
			ss << "DEPRECATED_BEHAVIOR";
			break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
			ss << "UNDEFINED_BEHAVIOR";
			break;
		case GL_DEBUG_TYPE_PORTABILITY:
			ss << "PORTABILITY";
			break;
		case GL_DEBUG_TYPE_PERFORMANCE:
			ss << "PERFORMANCE";
			break;
		case GL_DEBUG_TYPE_OTHER:
			ss << "OTHER";
			return;
			break;
		}
		ss << std::endl;

		ss << "id: " << id << std::endl;
		ss << "severity: ";
		switch (severity)
		{
		case GL_DEBUG_SEVERITY_LOW:
			ss << "LOW";
			break;
		case GL_DEBUG_SEVERITY_MEDIUM:
			ss << "MEDIUM";
			break;
		case GL_DEBUG_SEVERITY_HIGH:
			ss << "HIGH";
			break;
		}
		ss << std::endl;
		ss << "---------------------opengl-callback-end--------------" << std::endl;

		std::cout << ss.str() << std::endl;
//		if (isError) { _getch(); }
	}
};