#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "common/realtime.h"
#include "common/stopwatch.h"

#include "opengl/buffer.h"
#include "opengl/shader.h"
#include "opengl/query.h"

//#include "realtimetechniques/rtoptixutil.h"
#include <cuda.h>

#include "optix.h"
#include "optixu/optixu.h"
#include "optixu/optixpp_namespace.h"
#include "optix_gl_interop.h"

#include "../rtcommon.h"
#include "../rttechnique.h"

// An array of 3 vectors which represents 3 vertices
static const GLfloat gBigTriangleVertices[] =
{
-1.0f, -1.0f, 0.0f,
3.0f, -1.0f, 0.0f,
-1.0f,  3.0f, 0.0f,
};

#define USE_OPTIX_VPL

class RtPt2 : public RtTechnique
{
public:
	/// TODO::implement a proper constructor (if we have time)
	RtPt2()
	{
	}

	optix::float3 make_float3(const glm::vec3 & v)
	{
		optix::float3 result;
		result.x = v.x;
		result.y = v.y;
		result.z = v.z;
		return result;
	}

	enum EFrame
	{
		Accumulate = 1,
		ClearEveryFrame = 2
	};

	static void CheckFramebuffer()
	{
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE)
		{
			//std::cout << glewGetErrorString(status) << std::endl;
			switch (status)
			{
			case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
				std::cout << "incomplete layer targets" << std::endl;
			case GL_FRAMEBUFFER_UNSUPPORTED:
				std::cout << "unsupported" << std::endl;
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
				std::cout << "incomplete attachment" << std::endl;
				break;
			default:
				std::cout << "super error" << std::endl;
				break;
			}
			throw std::exception(); // "errorororororor"
		}
	}

	static const std::map<std::string, EFrame> EFrameModeStrMap;

	void render(shared_ptr<RtScene> & rtScene, const glm::vec2 & resolution, const nlohmann::json & json) override
	{
		mScene = rtScene;
		mCameraPosition = mScene->mCamera->getOrigin();
		mResolution = resolution;
		mInvResolution = glm::vec2(1.0f) / resolution;

		mRngOffset = json["rngOffset"];

		// deal with json file
		mNumMaxIteration = json["numMaxIteration"];
		mTimelimitMs = json["timeLimitMs"];

		mFrameMode = RtPt2::EFrameModeStrMap.at(json["frameMode"]);

		std::string p1 = json["outputFilename"];
		mOutputFilename = p1;

		std::string p2 = json["statFilename"];
		mStatFilename = p2;

		mJitter = json["useJitter"];
		mUseStat = json["useStat"];

		// serious parameters
		mNumSamplePerPixel = json["numSamplePerPixel"];
		mNumMaxBounce = json["numMaxBounces"];
		mDoWriteEveryFrame = (json.find("writeEveryFrame") == json.end()) ? false : json["writeEveryFrame"];

		this->setup();
		this->run();
		this->destroy();
	}

	FloatImage dumpImage(const glm::uvec2 & resolution, const std::function<void(void)> & renderFunc)
	{
		GLuint dumpFramebuffer;
		glGenFramebuffers(1, &dumpFramebuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, dumpFramebuffer);

		GLuint dumpTexture;
		glGenTextures(1, &dumpTexture);
		glBindTexture(GL_TEXTURE_2D, dumpTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mResolution.x, mResolution.y, 0, GL_RGBA, GL_FLOAT, 0);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, dumpTexture, 0);

		renderFunc();

		FloatImage image(resolution);
		glReadPixels(0, 0, resolution.x, resolution.y, GL_RGB, GL_FLOAT, (GLvoid *)(&(image._mData[0].x)));
		return image;
	}

	GLuint mDeferredFramebuffer;
	GLuint mDeferredDiffuse;
	GLuint mDeferredPhongReflectance;
	GLuint mDeferredNormal;
	GLuint mDeferredPosition;
	GLuint mDeferredDepthRenderbuffer;
	GLuint mDeferredLight;
	void setupDeferredTextures()
	{
		glGenFramebuffers(1, &mDeferredFramebuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, mDeferredFramebuffer);

		// position texture
		glGenTextures(1, &mDeferredPosition);
		glBindTexture(GL_TEXTURE_2D, mDeferredPosition);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mResolution.x, mResolution.y, 0, GL_RGBA, GL_FLOAT, 0);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mDeferredPosition, 0);

		// normal texture
		glGenTextures(1, &mDeferredNormal);
		glBindTexture(GL_TEXTURE_2D, mDeferredNormal);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mResolution.x, mResolution.y, 0, GL_RGBA, GL_FLOAT, 0);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, mDeferredNormal, 0);

		// diffuse texture
		glGenTextures(1, &mDeferredDiffuse);
		glBindTexture(GL_TEXTURE_2D, mDeferredDiffuse);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mResolution.x, mResolution.y, 0, GL_RGBA, GL_FLOAT, 0);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, mDeferredDiffuse, 0);

		// phong reflectance texture
		glGenTextures(1, &mDeferredPhongReflectance);
		glBindTexture(GL_TEXTURE_2D, mDeferredPhongReflectance);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mResolution.x, mResolution.y, 0, GL_RGBA, GL_FLOAT, 0);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, mDeferredPhongReflectance, 0);

		// depth
		GLuint depthRenderbuffer;
		glGenRenderbuffers(1, &depthRenderbuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, mResolution.x, mResolution.y);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);
		mDeferredDepthRenderbuffer = depthRenderbuffer;

		GLenum drawBuffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
		glDrawBuffers(4, drawBuffers);
		CheckFramebuffer();
	}

	shared_ptr<OpenglProgram> mDeferredProgram;
	shared_ptr<OpenglUniform> mDeferredProgram_uMvp;
	shared_ptr<OpenglUniform> mDeferredProgram_uDiffuse;
	shared_ptr<OpenglUniform> mDeferredProgram_uLambertReflectance;
	shared_ptr<OpenglUniform> mDeferredProgram_uPhongReflectance;
	shared_ptr<OpenglUniform> mDeferredProgram_uPhongExponent;
	void initDeferredProgram()
	{
		mDeferredProgram = make_shared<OpenglProgram>();
		mDeferredProgram->attachShader(OpenglVertexShader::CreateFromFile("shaders/deferred.vert"));
		mDeferredProgram->attachShader(OpenglGeometryShader::CreateFromFile("shaders/deferred.geom"));
		mDeferredProgram->attachShader(OpenglFragmentShader::CreateFromFile("shaders/deferred.frag"));
		mDeferredProgram->compile();

		mDeferredProgram_uMvp = mDeferredProgram->registerUniform("uMVP");
		mDeferredProgram_uDiffuse = mDeferredProgram->registerUniform("uDiffuse");
		mDeferredProgram_uLambertReflectance = mDeferredProgram->registerUniform("uLambertReflectance");
		mDeferredProgram_uPhongReflectance = mDeferredProgram->registerUniform("uPhongReflectance");
		mDeferredProgram_uPhongExponent = mDeferredProgram->registerUniform("uPhongExponent");
	}

	GLuint mBigTriangleBuffer;
	void setupBigTriangle()
	{
		glGenBuffers(1, &mBigTriangleBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, mBigTriangleBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(gBigTriangleVertices), gBigTriangleVertices, GL_STATIC_DRAW);
	}

	GLuint mLightTexture;
	GLuint mLightFramebuffer;
	void setupLightFramebuffer(GLuint depthRenderbuffer)
	{
		glGenFramebuffers(1, &mLightFramebuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, mLightFramebuffer);

		// position texture
		glGenTextures(1, &mLightTexture);
		glBindTexture(GL_TEXTURE_2D, mLightTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, mResolution.x, mResolution.y, 0, GL_RGB, GL_FLOAT, 0);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mLightTexture, 0);

		// depth
		glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, mResolution.x, mResolution.y);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);

		GLenum drawBuffers[] = {GL_COLOR_ATTACHMENT0};
		glDrawBuffers(1, drawBuffers);
		assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	}

	// render light source
	shared_ptr<OpenglProgram> mLightProgram;
	shared_ptr<OpenglUniform> mLightProgram_uLightIntensity;
	shared_ptr<OpenglUniform> mLightProgram_uMVP;
	void initLightProgram()
	{
		std::cout << "init render light source program" << std::endl;
		mLightProgram = make_shared<OpenglProgram>();
		mLightProgram->attachShader(OpenglVertexShader::CreateFromFile("shaders/light.vert"));
		mLightProgram->attachShader(OpenglFragmentShader::CreateFromFile("shaders/light.frag"));
		mLightProgram->compile();

		mLightProgram_uLightIntensity = mLightProgram->registerUniform("uLightIntensity");
		mLightProgram_uMVP = mLightProgram->registerUniform("uMVP");
	}

	shared_ptr<OpenglProgram> mFinalProgram;
	shared_ptr<OpenglUniform> mFinalProgram_uVplTextureScalingFactor;
	shared_ptr<OpenglUniform> mFinalProgram_uPhotonTexture;
	shared_ptr<OpenglUniform> mFinalProgram_uPhotonScalingFactor;
	shared_ptr<OpenglUniform> mFinalProgram_uLightTexture;
	shared_ptr<OpenglUniform> mFinalProgram_uLightScalingFactor;
	shared_ptr<OpenglUniform> mFinalProgram_uVplTextureBuffer;
	shared_ptr<OpenglUniform> mFinalProgram_uScreenWidth;
	shared_ptr<OpenglUniform> mFinalProgram_uDoGammaCorrection;
	void initFinalProgram()
	{
		mFinalProgram = make_shared<OpenglProgram>();
		mFinalProgram->attachShader(OpenglVertexShader::CreateFromFile("shaders/final.vert"));
		mFinalProgram->attachShader(OpenglFragmentShader::CreateFromFile("shaders/final.frag"));
		mFinalProgram->compile();

		mFinalProgram_uVplTextureScalingFactor = mFinalProgram->registerUniform("uVplScale");
		mFinalProgram_uPhotonTexture = mFinalProgram->registerUniform("uPhotonTexture");
		mFinalProgram_uPhotonScalingFactor = mFinalProgram->registerUniform("uPhotonScale");
		mFinalProgram_uLightTexture = mFinalProgram->registerUniform("uLightTexture");
		mFinalProgram_uLightScalingFactor = mFinalProgram->registerUniform("uLightScale");

		mFinalProgram_uVplTextureBuffer = mFinalProgram->registerUniform("uVplTextureBuffer");
		mFinalProgram_uScreenWidth = mFinalProgram->registerUniform("uScreenWidth");
		mFinalProgram_uDoGammaCorrection = mFinalProgram->registerUniform("uDoGammaCorrection");
	}

	//----------------------
	//      OPTIX
	//----------------------
	optix::Context mOptixContext;
	void initOptixContext()
	{
		mOptixContext = optix::Context::create();
		mOptixContext->setRayTypeCount(2);
		mOptixContext->setEntryPointCount(1);
		mOptixContext->setStackSize(4640);

		#if 0
			mOptixContext->setExceptionEnabled(RT_EXCEPTION_ALL, true);
			mOptixContext->setPrintEnabled(true);
			mOptixContext->setPrintBufferSize(1024);

			optix::Program exceptionProgram = mOptixContext->createProgramFromPTXFile("ptxfiles/pathtracing.cu.ptx", "exception");
			mOptixContext->setExceptionProgram(0, exceptionProgram);
		#endif
	}

	optix::Program mOptixMeshIntersectProgram;
	optix::Program mOptixMeshBboxProgram;
	void initOptixMeshIntersectProgram()
	{
		mOptixMeshIntersectProgram = mOptixContext->createProgramFromPTXFile("ptxfiles/reflectcuts_generated_triangleintersect.cu.ptx", "meshFineIntersect");
		mOptixMeshBboxProgram = mOptixContext->createProgramFromPTXFile("ptxfiles/reflectcuts_generated_triangleintersect.cu.ptx", "meshBound");
	}

	optix::Program mOptixVplAnyHitProgram;
	optix::Program mOptixRtClosestProgram;
	optix::Material mOptixRtMaterial;
	void initOptixRtMaterial()
	{
		try
		{
			mOptixRtMaterial = mOptixContext->createMaterial();
			mOptixRtMaterial->setClosestHitProgram(0, mOptixContext->createProgramFromPTXFile("ptxfiles/reflectcuts_generated_pathtracing.cu.ptx", "rtMaterialClosestHit"));
			mOptixRtMaterial->setAnyHitProgram(1, mOptixContext->createProgramFromPTXFile("ptxfiles/reflectcuts_generated_pathtracing.cu.ptx", "rtMaterialAnyHit"));
		}
		catch (optix::Exception & e)
		{
			std::cout << e.what() << std::endl;
		}
	}

	GLuint mGlOptixVplResultTbo;
	GLuint mGlOptixVplResultTexture;

	optix::Buffer mOptixPtResult;
	optix::Program mOptixPtProgram;
	optix::GeometryGroup mOptixTopGeometryGroup;
	optix::TextureSampler mOptixDeferredPositionTextureSampler;
	optix::TextureSampler mOptixDeferredNormalTextureSampler;
	optix::TextureSampler mOptixDeferredDiffuseTextureSampler;
	optix::TextureSampler mOptixDeferredPhongReflectanceTextureSampler;
	void initOptixPtProgram()
	{
		//RTformat format = RTformat::RT_FORMAT_UNSIGNED_BYTE4;
		RTformat format = RTformat::RT_FORMAT_FLOAT4;

		// create shadow pbo
		size_t elementSize;
		mOptixContext->checkError(rtuGetSizeForRTformat(format, &elementSize));

		// create buffer
		glGenBuffers(1, &mGlOptixVplResultTbo);
		glBindBuffer(GL_TEXTURE_BUFFER, mGlOptixVplResultTbo);
		glBufferData(GL_TEXTURE_BUFFER, elementSize * mResolution.x * mResolution.y, 0, GL_STREAM_DRAW);

		glGenTextures(1, &mGlOptixVplResultTexture);
		glBindTexture(GL_TEXTURE_BUFFER, mGlOptixVplResultTexture);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, mGlOptixVplResultTbo);

		// interop glbo
		try
		{
			mOptixPtResult = mOptixContext->createBufferFromGLBO(RT_BUFFER_OUTPUT, mGlOptixVplResultTbo);
			mOptixPtResult->setFormat(format);
			mOptixPtResult->setSize(mResolution.x, mResolution.y);

			// create vpl program
			mOptixPtProgram = mOptixContext->createProgramFromPTXFile("ptxfiles/reflectcuts_generated_pathtracing.cu.ptx", "splatColor");
			mOptixContext->setRayGenerationProgram(0, mOptixPtProgram);

			mOptixDeferredPositionTextureSampler = mOptixContext->createTextureSamplerFromGLImage(mDeferredPosition, RTgltarget::RT_TARGET_GL_TEXTURE_2D);
			mOptixDeferredDiffuseTextureSampler = mOptixContext->createTextureSamplerFromGLImage(mDeferredDiffuse, RTgltarget::RT_TARGET_GL_TEXTURE_2D);
			mOptixDeferredPhongReflectanceTextureSampler = mOptixContext->createTextureSamplerFromGLImage(mDeferredPhongReflectance, RTgltarget::RT_TARGET_GL_TEXTURE_2D);
			mOptixDeferredNormalTextureSampler = mOptixContext->createTextureSamplerFromGLImage(mDeferredNormal, RTgltarget::RT_TARGET_GL_TEXTURE_2D);
		}
		catch (const optix::Exception & e)
		{
			std::cout << e.what() << std::endl;
//			_getch();
		}
	}

	void setup()
	{
		rt.setup(mResolution);

		// disable AA 
		glDisable(GL_MULTISAMPLE);

		glEnable(GL_DEPTH_TEST);
		//glEnable(GL_CULL_FACE);
		glDepthFunc(GL_LEQUAL);

		glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);

		setupDeferredTextures();
		initDeferredProgram();

		setupBigTriangle();

		initFinalProgram();

		setupLightFramebuffer(mDeferredDepthRenderbuffer);
		initLightProgram();

		initOptixContext();
		initOptixRtMaterial();
		initOptixMeshIntersectProgram();
		initOptixPtProgram();

		for (size_t i = 0;i < mScene->mMaterials.size();i++)
		{
			mScene->mMaterials[i]->createOpenglTextures();
			mScene->mMaterials[i]->createOptixTextures(mOptixContext);
		}

		// put all geometry inside a one group
		mOptixTopGeometryGroup = mOptixContext->createGeometryGroup();
		for (size_t i = 0;i < mScene->mMeshes.size();i++)
		{
			mScene->mMeshes[i]->createOpenglBuffer();
			mScene->mMeshes[i]->createOptixMeshBuffer(mOptixContext);
			//mScene->mMeshes[i].createOptixMeshBufferFromOpenglBuffer(mOptixContext); // slow
			mScene->mMeshes[i]->createOptixGeometry(mOptixContext, mOptixMeshIntersectProgram, mOptixMeshBboxProgram, mOptixRtMaterial, *mScene->mMaterials[mScene->mMeshes[i]->mMatIndex]);
			mOptixTopGeometryGroup->addChild(mScene->mMeshes[i]->mOptix.mGeometryInstance);
		}

		mScene->mArealight->createOptixCdf(mOptixContext);

		// setup area light
		mOptixContext["areaLightVertices"]->set(mScene->mArealight->mMesh->mOptix.mVertices);
		mOptixContext["areaLightNormals"]->set(mScene->mArealight->mMesh->mOptix.mNormals);
		mOptixContext["areaLightIndices"]->set(mScene->mArealight->mMesh->mOptix.mIndices);
		mOptixContext["areaLightCdf"]->set(mScene->mArealight->mOptixCdfBuffer);
		mOptixContext["areaLightIntensity"]->set4fv(&(mScene->mArealight->mPrecomputedLightIntensity[0]));
		mOptixContext["areaLightArea"]->set1fv(&mScene->mArealight->mMeshArea);

		optix::Acceleration accel = mOptixContext->createAcceleration("Trbvh");
		accel->markDirty();
		mOptixTopGeometryGroup->setAcceleration(accel);
	}

	void runDeferredProgram(const glm::mat4 & mvpMatrix, const glm::mat4 & originalMvpMatrix)
	{
		glUseProgram(mDeferredProgram->mHandle);

		std::vector<shared_ptr<RtMesh>> rtMeshes = mScene->mMeshes;
		std::vector<shared_ptr<RtMaterial>> rtMaterials = mScene->mMaterials;

		for (size_t i = 0;i < rtMeshes.size();i++)
		{
			if (rtMeshes[i] == mScene->mArealight->mMesh)
			{
				mDeferredProgram_uMvp->setUniform(originalMvpMatrix);
			}
			else
			{
				mDeferredProgram_uMvp->setUniform(mvpMatrix);
			}

			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, rtMeshes[i]->mGl.mVerticesBuffer->mHandle);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);

			glEnableVertexAttribArray(1);
			glBindBuffer(GL_ARRAY_BUFFER, rtMeshes[i]->mGl.mTexCoordsBuffer->mHandle);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*) 0);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, rtMaterials[rtMeshes[i]->mMatIndex]->mLambertReflectance->mGlHandle);
			mDeferredProgram_uLambertReflectance->setUniform(0);

			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, rtMaterials[rtMeshes[i]->mMatIndex]->mPhongReflectance->mGlHandle);
			mDeferredProgram_uPhongReflectance->setUniform(1);

			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, rtMaterials[rtMeshes[i]->mMatIndex]->mPhongExponent->mGlHandle);
			mDeferredProgram_uPhongExponent->setUniform(2);

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rtMeshes[i]->mGl.mIndicesBuffer->mHandle);
			glDrawElements(GL_TRIANGLES, rtMeshes[i]->mNumTriangles * 3, GL_UNSIGNED_INT, (void*) 0);
			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);
		}
	}

	void runFinalProgram(const float ptScaling, const float lightScaling, const bool doGammaCorrection)
	{
		glUseProgram(mFinalProgram->mHandle);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, mBigTriangleBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glActiveTexture(GL_TEXTURE0);

		glBindTexture(GL_TEXTURE_BUFFER, mGlOptixVplResultTexture);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, mGlOptixVplResultTbo);
		mFinalProgram_uVplTextureBuffer->setUniform(0);
		mFinalProgram_uScreenWidth->setUniform((int)mResolution.x);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, mLightTexture);
		mFinalProgram_uLightTexture->setUniform(1);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, mLightTexture);
		mFinalProgram_uPhotonTexture->setUniform(2);

		mFinalProgram_uVplTextureScalingFactor->setUniform(ptScaling);
		mFinalProgram_uPhotonScalingFactor->setUniform(0.0f);
		mFinalProgram_uLightScalingFactor->setUniform(lightScaling);
		mFinalProgram_uDoGammaCorrection->setUniform(doGammaCorrection);

		glDrawArrays(GL_TRIANGLES, 0, 3); 

		glDisableVertexAttribArray(0);
	}

	void runLightProgram(const glm::mat4 & mvpMatrix)
	{
		glUseProgram(mLightProgram->mHandle);

		glEnableVertexAttribArray(0);
		RtMesh & areaLightMesh = *(mScene->mArealight->mMesh);
		const glm::vec3 & lightIntensity = mScene->mArealight->mLightIntensity;

		// check material on each trianglemesh
		mLightProgram_uLightIntensity->setUniform(lightIntensity);
		mLightProgram_uMVP->setUniform(mvpMatrix);

		glBindBuffer(GL_ARRAY_BUFFER, areaLightMesh.mGl.mVerticesBuffer->mHandle);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, areaLightMesh.mGl.mIndicesBuffer->mHandle);
		glDrawElements(GL_TRIANGLES, areaLightMesh.mNumTriangles * 3, GL_UNSIGNED_INT, (void*)0);

		glDisableVertexAttribArray(0);
	}

	void runOptixPtProgram(unsigned int rngSeed)
	{
		try
		{
			mOptixContext["rngSeed"]->setUint(rngSeed);
			// launching optix several times is faster than launching once and compute many spp on optix kernel.
			mOptixContext->launch(0, mResolution.x, mResolution.y);
		}
		catch (const optix::Exception & e)
		{
			std::cout << e.what() << std::endl;
		}
	}

	void run()
	{
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

		unique_ptr<Sampler> mainSampler = make_unique<IndependentSampler>(mRngOffset);

		glBindFramebuffer(GL_FRAMEBUFFER, mLightFramebuffer);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		int numIterations = 0;
		
		StopWatch masterWatch;
		masterWatch.reset();

		mOptixContext["cameraPosition"]->set3fv(&(mCameraPosition[0]));
		mOptixContext["outputBuffer"]->setBuffer(mOptixPtResult);
		mOptixContext["deferredPositionTexture"]->setTextureSampler(mOptixDeferredPositionTextureSampler);
		mOptixContext["deferredNormalTexture"]->setTextureSampler(mOptixDeferredNormalTextureSampler);
		mOptixContext["deferredDiffuseTexture"]->setTextureSampler(mOptixDeferredDiffuseTextureSampler);
		mOptixContext["deferredPhongReflectanceTexture"]->setTextureSampler(mOptixDeferredPhongReflectanceTextureSampler);
		mOptixContext["topObject"]->set(mOptixTopGeometryGroup);
		mOptixContext["boundingValue"]->setFloat(0);
		mOptixContext["maxBounces"]->setUint(mNumMaxBounce);

		if (mFrameMode == EFrame::ClearEveryFrame)
		{
			mOptixContext["doAccumulate"]->setUint(0);
		}
		else
		{
			mOptixContext["doAccumulate"]->setUint(1);
		}

		rt.loop([&] (std::string * extendString) // before swap buffer
		{
			if (numIterations == mNumMaxIteration) // write file if there's maximum number of iterations
			{
				return false;
			}

			glm::mat4 originalMvpMatrix = mScene->mCamera->computeVpMatrix();
			glm::mat4 mvpMatrix = originalMvpMatrix;

			if (mJitter)
			{
				// ndc coordinates = ([-1, 1], [-1, 1])
				glm::vec2 jitter = (2.0f * mainSampler->nextVec2() - glm::vec2(1)) * mInvResolution;
				glm::mat4 jitterMatrix = glm::translate(glm::vec3(jitter.x, jitter.y, 0));
				mvpMatrix = jitterMatrix * mvpMatrix;
			}

			// DEFERRED SHADING (~5-8% of 1 frame time for all scenes)
			glBindFramebuffer(GL_FRAMEBUFFER, mDeferredFramebuffer);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			runDeferredProgram(mvpMatrix, originalMvpMatrix);

			runOptixPtProgram(numIterations + mRngOffset);

			#if 1
				// RENDER LIGHT SOURCE
				{
				    // we don't jitter light source
					glBindFramebuffer(GL_FRAMEBUFFER, mLightFramebuffer);
					if (mFrameMode == EFrame::ClearEveryFrame)
					{
						glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
					}
					runLightProgram(originalMvpMatrix);
				}
			#endif

			#if 1
				// RENDER TEXTURE TO SCREEN
				{
					glBindFramebuffer(GL_FRAMEBUFFER, NULL);
					glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
					if (mFrameMode == EFrame::ClearEveryFrame)
					{
						runFinalProgram(1.0f, 1.0f, true);
					}
					else
					{
						runFinalProgram(1.0f / (float)(numIterations + 1), 1.0f, true);
					}
				}
			#endif

			numIterations++;

			return true;
		},
		[&](std::string * titleExtend) {
			if (masterWatch.timeMilliSec() >= mTimelimitMs) { return false; }

			if (mDoWriteEveryFrame)
			{
				FloatImage result;
				if (mFrameMode == EFrame::ClearEveryFrame)
				{
					result = dumpImage(this->mResolution, [&]() { runFinalProgram(1.0f, 1.0f, false); } );
				}
				else
				{
					FloatImage lightImage = dumpImage(this->mResolution, [&] () { runFinalProgram(0.0f, 1.0f, false); });
					FloatImage ptImage = dumpImage(this->mResolution, [&] () { runFinalProgram(1.0f, 0.0f, false); });
					ptImage /= (float) numIterations;
					result = lightImage + ptImage;
				}

				size_t i = mOutputFilename.find_last_of('.');
				assert(i > 0 && i < mOutputFilename.length() - 1);
				std::string dotExtension = mOutputFilename.substr(i);
				
				FloatImage::Save(FloatImage::FlipY(result), mOutputFilename.substr(0, i) + "_" + std::to_string(numIterations) + dotExtension);
			}

			return true;
		});

		if (mUseStat)
		{
			// write stat
			nlohmann::json result;
			std::cout << masterWatch.timeMilliSec() << std::endl;
			result["time"] = masterWatch.timeMilliSec();
			result["numIterations"] = numIterations;
			std::ofstream of(mStatFilename);
			assert(of.is_open());
			of << std::setw(4) << result;
		}

		FloatImage result;
		if (mFrameMode == EFrame::ClearEveryFrame)
		{
			result = dumpImage(this->mResolution, [&]() { runFinalProgram(1.0f, 1.0f, false); } );
		}
		else
		{
			FloatImage lightImage = dumpImage(this->mResolution, [&] () { runFinalProgram(0.0f, 1.0f, false); });
			FloatImage ptImage = dumpImage(this->mResolution, [&] () { runFinalProgram(1.0f, 0.0f, false); });
			ptImage /= (float) numIterations;
			result = lightImage + ptImage;
		}

		FloatImage::Save(FloatImage::FlipY(result), mOutputFilename);
	}

	void destroy()
	{
		rt.destroy();
	}

	glm::vec3 mCameraPosition;
	glm::uvec2 mResolution;
	glm::vec2 mInvResolution;
	float mTimelimitMs;

	int mNumMaxIteration = 0;
	EFrame mFrameMode;
	unsigned int mRngOffset = 0;

	int mNumMaxBounce = 0;
	int mNumSamplePerPixel = 0;

	// number of vpl used in each frame
	bool mJitter;
	bool mUseStat;

	bool mDoWriteEveryFrame = false;

	// wait for atleast n frames before gpu start to work properly
	int mColdstartFrames = 2;

	std::string mOutputFilename;
	std::string mStatFilename;
	shared_ptr<RtScene> mScene;

	RealTime rt;
};

const std::map<std::string, RtPt2::EFrame> RtPt2::EFrameModeStrMap = {
	{ "accumulate", RtPt2::EFrame::Accumulate },
	{ "cleareveryframe", RtPt2::EFrame::ClearEveryFrame }
};