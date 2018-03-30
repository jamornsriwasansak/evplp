#pragma once

#include "common/reflectcuts.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "common/realtime.h"
#include "common/stopwatch.h"
#include "shapes/trianglemesh.h"

#include "opengl/buffer.h"
#include "opengl/shader.h"
#include "opengl/query.h"

#include <cuda.h>

#include "optix.h"
#include "optixu/optixu.h"
#include "optixu/optixpp_namespace.h"
#include "optix_gl_interop.h"

#include "../rtcommon.h"
#include "../rttechnique.h"

#include "rtphotonrecord.h"

#define USE_OPTIX_VPL

// Implementation of EVPLP (Energy Compesated VPL using Photons)
class RtComPhoton: public RtTechnique
{
public:
	/// TODO::implement a proper constructor (if we have time)
	RtComPhoton()
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

	static const std::map<std::string, EFrame> EFrameModeStrMap;

	enum ERender
	{
		Vpl = 1 << 0,
		Photon = 1 << 1,
		VplPhoton = 1 << 2,
		None = 1 << 3
	};

	enum EMis
	{
		One = 0,
		Balance = 1,
		Max = 2,
		Power2 = 3,
		GeometryClamp = 4,
		GeometryBrdfClamp = 5
	};

	static const std::map<std::string, EMis> EMisModeStrMap;

	enum EOptixPasses
	{
		LightTrace,
		VplSplat,
		NumPass,
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

	void render(shared_ptr<RtScene> & scene, const glm::vec2 & resolution, const nlohmann::json & json) override
	{
		mScene = scene;
		mResolution = resolution;
		mInvResolution = glm::vec2(1.0f) / resolution;

		// serious parameters
		mNumLightPaths = json["numLightPaths"];
		mNumVplLightPaths = json["numVplLightPaths"];
		mNumMaxBounce = json["numMaxBounces"];
		mNumPhotonsPerLightPath = mNumMaxBounce + 1;
		mRadiusPercentage = json["radiusPercentage"];
		mPhotonRadius = mScene->findBoundingSphereRadius() * mRadiusPercentage;
		mPrecomptedPdfMc = static_cast<float>(mNumVplLightPaths) / static_cast<float>(mNumLightPaths) * Math::InvPi / (mPhotonRadius * mPhotonRadius);
		mDoWriteEveryFrame = (json.find("writeEveryFrame") == json.end()) ? false : json["writeEveryFrame"];

		// deal with json file
		mNumMaxIteration = json["numMaxIteration"];
		mTimelimitMs = json["timeLimitMs"];

		mFrameMode = RtComPhoton::EFrameModeStrMap.at(json["frameMode"]);
		if(json.find("misMode") == json.end())
		{
			mMisMode = RtComPhoton::EMis::Balance;
		}
		else
		{
			mMisMode = RtComPhoton::EMisModeStrMap.at(json["misMode"]);
		}

		if (json.find("clampingStart") != json.end())
		{
			std::cerr << "clampingStart option is not use anymore\n";
			std::cerr << "remove it from your JSON file\n";
			assert(false);
		}

		if (json.find("targetRenderingTime") != json.end()) {
			mTargetRenderingTime = json["targetRenderingTime"];
		}

		if(json.find("clampingCoeff") == json.end())
		{
			// Automatic clamping value ...
			float totalArea = scene->totalArea();
			std::cout << "Total area computation: " << totalArea << "\n";
			mClampingValue = 1.f / totalArea;
			mClampingStart = 1.f / totalArea;
		}
		else
		{
			float clampCoeff = json["clampingCoeff"];
			mClampingValue = clampCoeff;
			mClampingStart = clampCoeff;
		}

		mRngOffset = json["rngOffset"];

		std::string p1 = json["combinedFilename"];
		mDumpCombineFilename = p1;

		std::string p2 = json["weightedPhotonFilename"];
		mDumpWeightedPhotonFilename = p2;

		std::string p3 = json["weightedVplFilename"];
		mDumpWeightedVplFilename = p3;

		std::string p4 = json["statFilename"];
		mStatFilename = p4;
		
		mJitter = json["useJitter"];
		mUseStat = json["useStat"];

		if (json.find("DoProgressive") != json.end()) {
			mDoProgressive = json["DoProgressive"];
		}

		if (json.find("AlphaProgressive") != json.end()) {
			mAlphaProgressive = json["AlphaProgressive"];
		}

		if (json.find("run") != json.end())
		{
			const nlohmann::json & runJson = json["run"];
			if (runJson.find("deferredShading") != runJson.end()) { mDoDeferredShading = runJson["deferredShading"]; }
			if (runJson.find("lightTracing") != runJson.end()) { mDoLightTracing = runJson["lightTracing"]; }
			if (runJson.find("vplSplat") != runJson.end()) { mDoVplSplat = runJson["vplSplat"]; }
			if (runJson.find("photonSplat") != runJson.end()) { mDoPhotonSplat = runJson["photonSplat"]; }
			if (runJson.find("lightRender") != runJson.end()) { mDoLightRender= runJson["lightRender"]; }
			if (runJson.find("finalize") != runJson.end()) { mDoFinalize = runJson["finalize"]; }
		}

		// Change the render mode depending on 
		if (mNumVplLightPaths == 0) {
			std::cout << "WARN: 0 VPL light paths. Disable mDoVplSplat\n";
			mDoVplSplat = false;
		}

		if (json.find("forceVsl") != json.end()) {
			mForceVsl = json["forceVsl"];
			if (mForceVsl)
			{
				mVslRadiusPercentage = json["vslRadiusPercentage"];
				mVslRadius  = mScene->findBoundingSphereRadius() * mVslRadiusPercentage;
				if (mVslRadius <= 0.008)
				{
					mVslRadius = std::max(mVslRadius, 0.008f);
					std::cout << "warning : vslRadius is too small. clamped vslRadius" << std::endl;
				}
				mVslInvPiRadius2 = Math::InvPi / (mVslRadius * mVslRadius);
			}
		}

		setup();
		run();
		destroy();
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

		glDeleteFramebuffers(1, &dumpFramebuffer);
		glDeleteTextures(1, &dumpTexture);
		return image;
	}

	GLuint mDeferredFramebuffer;
	GLuint mDeferredDiffuse;
	GLuint mDeferredPhongInfo;
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

		// phong info texture
		glGenTextures(1, &mDeferredPhongInfo);
		glBindTexture(GL_TEXTURE_2D, mDeferredPhongInfo);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mResolution.x, mResolution.y, 0, GL_RGBA, GL_FLOAT, 0);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, mDeferredPhongInfo, 0);

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
		mOptixContext->setEntryPointCount(EOptixPasses::NumPass);
		mOptixContext->setStackSize(4640);

		#if 0
			mOptixContext->setExceptionEnabled(RT_EXCEPTION_ALL, true);
			mOptixContext->setPrintEnabled(true);
			mOptixContext->setPrintBufferSize(1024);
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
			mOptixRtMaterial->setClosestHitProgram(0, mOptixContext->createProgramFromPTXFile("ptxfiles/reflectcuts_generated_lighttracing.cu.ptx", "rtMaterialClosestHit"));
			mOptixRtMaterial->setAnyHitProgram(1, mOptixContext->createProgramFromPTXFile("ptxfiles/reflectcuts_generated_lighttracing.cu.ptx", "rtMaterialAnyHit"));
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
	optix::TextureSampler mOptixDeferredPhongInfoSampler;
	void initOptixVplProgram()
	{
		RTformat format = RTformat::RT_FORMAT_FLOAT4;
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
			if (mForceVsl)
			{
				mOptixPtProgram = mOptixContext->createProgramFromPTXFile("ptxfiles/reflectcuts_generated_lighttracing.cu.ptx", "splatSplotch");
			}
			else
			{
				mOptixPtProgram = mOptixContext->createProgramFromPTXFile("ptxfiles/reflectcuts_generated_lighttracing.cu.ptx", "splatColor");
			}

			mOptixContext->setRayGenerationProgram(EOptixPasses::VplSplat, mOptixPtProgram);

			optix::Program exceptionProgram = mOptixContext->createProgramFromPTXFile("ptxfiles/reflectcuts_generated_lighttracing.cu.ptx", "exception");
			mOptixContext->setExceptionProgram(EOptixPasses::VplSplat, exceptionProgram);

			mOptixDeferredPositionTextureSampler = mOptixContext->createTextureSamplerFromGLImage(mDeferredPosition, RTgltarget::RT_TARGET_GL_TEXTURE_2D);
			mOptixDeferredDiffuseTextureSampler = mOptixContext->createTextureSamplerFromGLImage(mDeferredDiffuse, RTgltarget::RT_TARGET_GL_TEXTURE_2D);
			mOptixDeferredNormalTextureSampler = mOptixContext->createTextureSamplerFromGLImage(mDeferredNormal, RTgltarget::RT_TARGET_GL_TEXTURE_2D);
			mOptixDeferredPhongInfoSampler = mOptixContext->createTextureSamplerFromGLImage(mDeferredPhongInfo, RTgltarget::RT_TARGET_GL_TEXTURE_2D);
		}
		catch (const optix::Exception & e)
		{
			std::cout << e.what() << std::endl;
//			_getch();
		}
	}

	GLuint mOptixPhotonSsboHandle;
	optix::Buffer mOptixPhotonRecordsBuffer;
	optix::Buffer mOptixPhotonInfoBuffer;
	optix::Program mOptixLightTracingProgram;
	void initOptixLightTracingProgram()
	{
		// create ssbo
		glGenBuffers(1, &mOptixPhotonSsboHandle);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, mOptixPhotonSsboHandle);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(RtPhotonRecord) * mNumPhotonsPerLightPath * mNumLightPaths, NULL, GL_DYNAMIC_COPY);

		try
		{
			// create optix buffer
			mOptixPhotonRecordsBuffer = mOptixContext->createBufferFromGLBO(RT_BUFFER_OUTPUT, mOptixPhotonSsboHandle);
			mOptixPhotonRecordsBuffer->setFormat(RT_FORMAT_USER);
			mOptixPhotonRecordsBuffer->setElementSize(sizeof(RtPhotonRecord));
			mOptixPhotonRecordsBuffer->setSize(mNumLightPaths * mNumPhotonsPerLightPath);

			mOptixPhotonInfoBuffer = mOptixContext->createBuffer(RT_BUFFER_INPUT_OUTPUT);
			mOptixPhotonInfoBuffer->setFormat(RT_FORMAT_USER);
			mOptixPhotonInfoBuffer->setElementSize(sizeof(RtPhotonInfo));
			mOptixPhotonInfoBuffer->setSize(1);

			mOptixLightTracingProgram = mOptixContext->createProgramFromPTXFile("ptxfiles/reflectcuts_generated_lighttracing.cu.ptx", "tracePhotons");
			mOptixContext->setRayGenerationProgram(EOptixPasses::LightTrace, mOptixLightTracingProgram);

			optix::Program exceptionProgram = mOptixContext->createProgramFromPTXFile("ptxfiles/reflectcuts_generated_lighttracing.cu.ptx", "exception");
			mOptixContext->setExceptionProgram(EOptixPasses::LightTrace, exceptionProgram);
		}
		catch (const optix::Exception & e)
		{
			std::cout << e.what() << std::endl;
//			_getch();
		}
	}


	// ** reuse depth buffer from deferred pass
	GLuint mPhotonSplatTexture;
	GLuint mPhotonSplatFramebuffer;
	void setupPhotonSplatFramebuffer(GLuint depthRenderbuffer)
	{
		glGenFramebuffers(1, &mPhotonSplatFramebuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, mPhotonSplatFramebuffer);

		// position texture
		glGenTextures(1, &mPhotonSplatTexture);
		glBindTexture(GL_TEXTURE_2D, mPhotonSplatTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, mResolution.x, mResolution.y, 0, GL_RGB, GL_FLOAT, 0);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mPhotonSplatTexture, 0);

		// depth
		glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, mResolution.x, mResolution.y);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);

		GLenum drawBuffers[] = {GL_COLOR_ATTACHMENT0};
		glDrawBuffers(1, drawBuffers);
		assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	}

	shared_ptr<OpenglProgram> mPhotonSplatProgram;
	shared_ptr<OpenglUniform> mPhotonSplatProgram_uPhotonRadius;
	shared_ptr<OpenglUniform> mPhotonSplatProgram_uMVP;
	shared_ptr<OpenglUniform> mPhotonSplatProgram_uPositionTexture;
	shared_ptr<OpenglUniform> mPhotonSplatProgram_uInvPhotonRadius2;
	shared_ptr<OpenglUniform> mPhotonSplatProgram_uInvNumLightPaths;
	shared_ptr<OpenglUniform> mPhotonSplatProgram_uNormalTexture;
	shared_ptr<OpenglUniform> mPhotonSplatProgram_uDiffuseTexture;
	shared_ptr<OpenglUniform> mPhotonSplatProgram_uPhongInfoTexture;
	shared_ptr<OpenglUniform> mPhotonSplatProgram_uCameraPosition;
	shared_ptr<OpenglUniform> mPhotonSplatProgram_uPdfMc;
	shared_ptr<OpenglUniform> mPhotonSplatProgram_uClampingValue;
	shared_ptr<OpenglUniform> mPhotonSplatProgram_uMisMode;
	GLuint mPhotonSplatProgram_uSsboBindingPointIndex;
	void initPhotonSplatProgram()
	{
		mPhotonSplatProgram = make_shared<OpenglProgram>();
		mPhotonSplatProgram->attachShader(OpenglVertexShader::CreateFromFile("shaders/photonsplatinstanced.vert"));
		mPhotonSplatProgram->attachShader(OpenglGeometryShader::CreateFromFile("shaders/photonsplatinstanced.geom"));
		mPhotonSplatProgram->attachShader(OpenglFragmentShader::CreateFromFile("shaders/photonsplatinstanced.frag"));
		mPhotonSplatProgram->compile();

		mPhotonSplatProgram_uMVP = mPhotonSplatProgram->registerUniform("uMVP");
		mPhotonSplatProgram_uPhotonRadius = mPhotonSplatProgram->registerUniform("uPhotonRadius");
		mPhotonSplatProgram_uInvPhotonRadius2 = mPhotonSplatProgram->registerUniform("uInvPhotonRadius2");
		mPhotonSplatProgram_uInvNumLightPaths = mPhotonSplatProgram->registerUniform("uInvNumLightPaths");
		
		mPhotonSplatProgram_uPositionTexture = mPhotonSplatProgram->registerUniform("uPositionTexture");
		mPhotonSplatProgram_uDiffuseTexture = mPhotonSplatProgram->registerUniform("uDiffuseTexture");
		mPhotonSplatProgram_uNormalTexture = mPhotonSplatProgram->registerUniform("uNormalTexture");
		mPhotonSplatProgram_uPhongInfoTexture = mPhotonSplatProgram->registerUniform("uPhongInfoTexture");
		mPhotonSplatProgram_uCameraPosition = mPhotonSplatProgram->registerUniform("uCameraPosition");
		mPhotonSplatProgram_uPdfMc = mPhotonSplatProgram->registerUniform("uPdfMc");
		mPhotonSplatProgram_uClampingValue = mPhotonSplatProgram->registerUniform("uClampingValue");
		mPhotonSplatProgram_uMisMode = mPhotonSplatProgram->registerUniform("uMisMode");

		GLuint blockIndex = glGetProgramResourceIndex(mPhotonSplatProgram->mHandle, GL_SHADER_STORAGE_BLOCK, "PhotonRecords");
		mPhotonSplatProgram_uSsboBindingPointIndex = 0;
		glShaderStorageBlockBinding(mPhotonSplatProgram->mHandle, blockIndex, mPhotonSplatProgram_uSsboBindingPointIndex);
	}

	struct Icosohedron
	{
		shared_ptr<OpenglBuffer> mVerticesBuffer;
		shared_ptr<OpenglBuffer> mIndicesBuffer;
		shared_ptr<OpenglBuffer> mTexCoordsBuffer;
		GLuint mNumIndices;
	} mIcosohedron;

	void setupPhotonSplatIcosohedron(const std::string & filepath)
	{
		std::vector<shared_ptr<TriangleMesh>> trimeshes = TriangleMesh::LoadMeshes(filepath);
		assert(trimeshes.size() == 1);

		TriangleMesh & icosohedronTrimesh = *(trimeshes[0].get());
		icosohedronTrimesh.uploadOpenglBuffer();

		mIcosohedron.mVerticesBuffer = icosohedronTrimesh.mVerticesBuffer;
		mIcosohedron.mIndicesBuffer = icosohedronTrimesh.mIndicesBuffer;
		mIcosohedron.mTexCoordsBuffer = icosohedronTrimesh.mTexCoordsBuffer;
		mIcosohedron.mNumIndices = icosohedronTrimesh.mTriangles.size();
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
		initOptixVplProgram();
		initOptixLightTracingProgram();

		initPhotonSplatProgram();
		setupPhotonSplatFramebuffer(mDeferredDepthRenderbuffer); // must do after setup deferred textures
		setupPhotonSplatIcosohedron("sphere/icosphere.obj");
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
		mDeferredProgram_uMvp->setUniform(mvpMatrix);

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

	void runFinalProgram(const float vplScaling, const float photonScaling, const float lightScaling, const bool doGammaCorrection)
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
		glBindTexture(GL_TEXTURE_2D, mPhotonSplatTexture);
		mFinalProgram_uPhotonTexture->setUniform(1);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, mLightTexture);
		mFinalProgram_uLightTexture->setUniform(2);

		mFinalProgram_uVplTextureScalingFactor->setUniform(vplScaling);
		mFinalProgram_uPhotonScalingFactor->setUniform(photonScaling);
		mFinalProgram_uLightScalingFactor->setUniform(lightScaling);
		mFinalProgram_uDoGammaCorrection->setUniform(doGammaCorrection);

		glDrawArrays(GL_TRIANGLES, 0, 3); 

		glDisableVertexAttribArray(0);
	}

	void runPhotonSplat(const float radius, const glm::mat4 & mvpMatrix)
	{
		// enable depth test but disable depth writing
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);
		glBlendEquation(GL_FUNC_ADD);
		glDepthMask(GL_FALSE);

		glUseProgram(mPhotonSplatProgram->mHandle);

		mPhotonSplatProgram_uMVP->setUniform(mvpMatrix);
		mPhotonSplatProgram_uPhotonRadius->setUniform(radius);
		mPhotonSplatProgram_uCameraPosition->setUniform(mScene->mCamera->getOrigin());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, mDeferredPosition);
		mPhotonSplatProgram_uPositionTexture->setUniform(0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, mDeferredNormal);
		mPhotonSplatProgram_uNormalTexture->setUniform(1);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, mDeferredDiffuse);
		mPhotonSplatProgram_uDiffuseTexture->setUniform(2);

		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, mDeferredPhongInfo);
		mPhotonSplatProgram_uPhongInfoTexture->setUniform(3);

		mPhotonSplatProgram_uInvPhotonRadius2->setUniform(1.0f / (radius * radius));
		mPhotonSplatProgram_uInvNumLightPaths->setUniform(1.0f / static_cast<float>(mNumLightPaths));
		mPhotonSplatProgram_uPdfMc->setUniform(mPrecomptedPdfMc);
		mPhotonSplatProgram_uClampingValue->setUniform(mClampingValue);
		mPhotonSplatProgram_uMisMode->setUniform(mMisMode);

		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, mPhotonSplatProgram_uSsboBindingPointIndex, mOptixPhotonSsboHandle);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, mIcosohedron.mVerticesBuffer->mHandle);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIcosohedron.mIndicesBuffer->mHandle);
		glDrawElementsInstanced(GL_TRIANGLES, mIcosohedron.mNumIndices * 3, GL_UNSIGNED_INT, (void*)0, mNumLightPaths * mNumPhotonsPerLightPath);
		glDisableVertexAttribArray(0);

		glDepthMask(GL_TRUE);
		glDisable(GL_BLEND);
	}

	void runLightProgram(const glm::mat4 & mvpMatrix)
	{
		glUseProgram(mLightProgram->mHandle);

		glEnableVertexAttribArray(0);

		mLightProgram_uLightIntensity->setUniform(glm::vec3(mScene->mArealight->mLightIntensity));
		mLightProgram_uMVP->setUniform(mvpMatrix);

		glBindBuffer(GL_ARRAY_BUFFER, mScene->mArealight->mMesh->mGl.mVerticesBuffer->mHandle);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mScene->mArealight->mMesh->mGl.mIndicesBuffer->mHandle);
		glDrawElements(GL_TRIANGLES, mScene->mArealight->mMesh->mNumTriangles * 3, GL_UNSIGNED_INT, (void*)0);

		glDisableVertexAttribArray(0);
	}

	void runOptixVplProgram()
	{
		try
		{
			mOptixContext->launch(EOptixPasses::VplSplat, mResolution.x, mResolution.y);
		}
		catch (const optix::Exception & e)
		{
			std::cout << e.what() << std::endl;
		}
	}

	void runOptixLightTracingProgram(unsigned int rngSeed)
	{
		try
		{
			mOptixContext["rngSeed"]->setUint(rngSeed);
			mOptixContext->launch(EOptixPasses::LightTrace, mNumLightPaths);
		}
		catch (const optix::Exception & e)
		{
			std::cout << e.what() << std::endl;
//			_getch();
		}
	}

	void run()
	{
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

		unique_ptr<Sampler> mainSampler = make_unique<IndependentSampler>(mRngOffset);

		glBindFramebuffer(GL_FRAMEBUFFER, mLightFramebuffer);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		int numIterations = 0;
		
		// vpl stuffs
		glm::vec3 origin = mScene->mCamera->getOrigin();
		mOptixContext["cameraPosition"]->set3fv(&(origin[0]));
		mOptixContext["outputBuffer"]->setBuffer(mOptixPtResult);
		mOptixContext["deferredPositionTexture"]->setTextureSampler(mOptixDeferredPositionTextureSampler);
		mOptixContext["deferredNormalTexture"]->setTextureSampler(mOptixDeferredNormalTextureSampler);
		mOptixContext["deferredDiffuseTexture"]->setTextureSampler(mOptixDeferredDiffuseTextureSampler);
		mOptixContext["deferredPhongReflectanceTexture"]->setTextureSampler(mOptixDeferredPhongInfoSampler);
		mOptixContext["topObject"]->set(mOptixTopGeometryGroup);
		mOptixContext["boundingValue"]->setFloat(0);
		mOptixContext["pdfMc"]->setFloat(mPrecomptedPdfMc);
		mOptixContext["radius"]->setFloat(mPhotonRadius);
		mOptixContext["misMode"]->setUint(mMisMode);
		mOptixContext["clampingValue"]->setFloat(mClampingValue);

		// light trace stuffs
		mOptixContext["photons"]->setBuffer(mOptixPhotonRecordsBuffer);
		mOptixContext["photonInfo"]->setBuffer(mOptixPhotonInfoBuffer);
		mOptixContext["topObject"]->set(mOptixTopGeometryGroup);
		mOptixContext["numVplLightPaths"]->setUint(mNumVplLightPaths);
		mOptixContext["numLightPaths"]->setUint(mNumLightPaths);
		mOptixContext["numPhotonsPerLightPath"]->setUint(mNumPhotonsPerLightPath);

		if (mForceVsl)
		{
			mOptixContext["vslInvPiRadius2"]->setFloat(mVslInvPiRadius2);
			mOptixContext["vslRadius"]->setFloat(mVslRadius);
		}

		if (mFrameMode == EFrame::ClearEveryFrame)
		{
			mOptixContext["doAccumulate"]->setUint(0);
		}
		else
		{
			mOptixContext["doAccumulate"]->setUint(1);
		}

		StopWatch masterWatch;
		masterWatch.reset();
		float prevTiming = 0.f;

		rt.loop([&](std::string * extendString) // before swap buffer
		{
			if(numIterations == mNumMaxIteration) // write file if there's maximum number of iterations
			{
				return false;
			}

			glm::mat4 originalMvpMatrix = mScene->mCamera->computeVpMatrix();
			glm::mat4 mvpMatrix = originalMvpMatrix;

			if(mJitter)
			{
				// ndc coordinates = ([-1, 1], [-1, 1])
				glm::vec2 jitter = (2.0f * mainSampler->nextVec2() - glm::vec2(1)) * mInvResolution;
				glm::mat4 jitterMatrix = glm::translate(glm::vec3(jitter.x, jitter.y, 0));
				mvpMatrix = jitterMatrix * mvpMatrix;
			}

			if (mDoDeferredShading)
			{
				// DEFERRED SHADING
				glBindFramebuffer(GL_FRAMEBUFFER, mDeferredFramebuffer);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				runDeferredProgram(mvpMatrix, originalMvpMatrix);
			}

			if (mDoLightTracing)
			{
				// LIGHT TRACING
				runOptixLightTracingProgram(numIterations + mRngOffset);
			}

			if (mDoVplSplat)
			{
				// VPL SPLATING
				runOptixVplProgram();
			}

			if (mDoPhotonSplat)
			{
				// PHOTON SPLATTING
				glBindFramebuffer(GL_FRAMEBUFFER, mPhotonSplatFramebuffer);
				if (mFrameMode == EFrame::ClearEveryFrame)
				{
					glClear(GL_COLOR_BUFFER_BIT);
				}
				runPhotonSplat(mPhotonRadius, mvpMatrix);
			}

			if (mDoLightRender)
			{
				// RENDER LIGHT SOURCE
				// we don't jitter light source
				glBindFramebuffer(GL_FRAMEBUFFER, mLightFramebuffer);
				if (mFrameMode == EFrame::ClearEveryFrame)
				{
					glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				}
				runLightProgram(originalMvpMatrix);
			}

			if (mDoFinalize)
			{
				// RENDER TEXTURE TO SCREEN
				glBindFramebuffer(GL_FRAMEBUFFER, NULL);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				float param = (mFrameMode == EFrame::ClearEveryFrame) ? 1.0f : (1.0f / (float)(numIterations + 1));
				runFinalProgram(param, param, 1.0f, true);
			}

			numIterations++;

			if (numIterations % 20 == 0) {
				// Jump dump time to time the number of iterations
				float currentTiming = masterWatch.timeMilliSec();
				float frameTime = currentTiming - prevTiming;
				frameTime /= 20;
				std::cout << "numIter: " << numIterations << " | raduis: " << mPhotonRadius << " | clamping: " << mClampingValue << " | timing: " << currentTiming - prevTiming << "\n";
				prevTiming = currentTiming;

				// Get the statistics about the rendering
				if (mTargetRenderingTime != -1) {
					float factor = mTargetRenderingTime / (frameTime);
					if (factor != 1.f) {
						std::cout << "change number of samples: " << factor << " | currFrame time: " << frameTime << "\n";
						if(mNumVplLightPaths != 0) {
							int newNbVPL = mNumVplLightPaths  * factor;
							std::cout << "Nb light paths: " << newNbVPL * (mNumLightPaths  / mNumVplLightPaths ) << "\n";
							std::cout << "Nb VPL paths: " << newNbVPL << "\n";
						}
						else {
							std::cout << "Nb light paths: " << mNumLightPaths * factor << "\n";
						}
					}
				}
			}

			if (mDoProgressive) {
				// Knaus and Zwiker start this numIteration == 1
				// So we need to update the kernel only after we have 
				// updated the numIteration
				Float ratio = (numIterations + mAlphaProgressive) / (numIterations + 1);
				mPhotonRadius *= std::sqrt(ratio);
				mClampingValue = mClampingStart * std::pow(numIterations, mAlphaProgressive); // Use for VPL clamping
				mPrecomptedPdfMc = static_cast<float>(mNumVplLightPaths) / static_cast<float>(mNumLightPaths) * Math::InvPi / (mPhotonRadius * mPhotonRadius);

				// Update the radius 
				mOptixContext["radius"]->setFloat(mPhotonRadius);
				mOptixContext["clampingValue"]->setFloat(mClampingValue);
				mOptixContext["pdfMc"]->setFloat(mPrecomptedPdfMc);

				if (mForceVsl)
				{
					mVslRadius *= std::sqrt(ratio);
					if (mVslRadius <= 0.008f)
					{
						mVslRadius = std::max(mVslRadius, 0.008f);
						std::cout << "warning : vslRadius is too small. clamped vslRadius" << std::endl;
					}

					mVslInvPiRadius2 = Math::InvPi / (mVslRadius * mVslRadius);
					if (numIterations % 20 == 0) {
						std::cout << "VSL Radius: " << mVslRadius << "\n";
					}
					mOptixContext["vslInvPiRadius2"]->setFloat(mVslInvPiRadius2);
					mOptixContext["vslRadius"]->setFloat(mVslRadius);
				}
			}

			if(masterWatch.timeMilliSec() >= mTimelimitMs) { return false; }

			return true;
		},
		[&](std::string * titleExtend) {
			if (mNumMaxIteration > 0)
			{
				float iterRatio = static_cast<float>(numIterations) / static_cast<float>(mNumMaxIteration);
				float time = static_cast<float>(masterWatch.timeMilliSec()) / 1000.0f;
				float etc = time / iterRatio - time;
				*titleExtend = std::to_string(iterRatio * 100.0f) + "%, ETC : " + std::to_string(etc) + "s";
			}


			if (mDoWriteEveryFrame)
			{
				FloatImage result;
				if (mFrameMode == EFrame::ClearEveryFrame)
				{
					result = dumpImage(this->mResolution, [&]() { runFinalProgram(1.0f, 1.0f, 1.0f, false); } );
				}
				else
				{
					float param = (mFrameMode == EFrame::ClearEveryFrame) ? 1.0f : (1.0f / (float)(numIterations));
					FloatImage lightSourceImage = dumpImage(this->mResolution, [&] () { runFinalProgram(0.0f, 0.0f, 1.0f, false); });
					FloatImage photonImage = dumpImage(this->mResolution, [&] () { runFinalProgram(0.0f, 1.0f, 0.0f, false); });
					photonImage *= param;
					FloatImage vplImage = dumpImage(this->mResolution, [&] () { runFinalProgram(1.0f, 0.0f, 0.0f, false); });
					vplImage *= param;
					result = lightSourceImage + photonImage + vplImage;
				}

				size_t i = mDumpWeightedPhotonFilename.find_last_of('.');
				assert(i > 0 && i < mDumpWeightedPhotonFilename.length() - 1);
				std::string dotExtension = mDumpWeightedPhotonFilename.substr(i);
				
				FloatImage::Save(FloatImage::FlipY(result), mDumpWeightedPhotonFilename.substr(0, i) + "_" + std::to_string(numIterations) + dotExtension);
			}

			return true;
		});

		float time = masterWatch.timeMilliSec();

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
		float param = (mFrameMode == EFrame::ClearEveryFrame) ? 1.0f : (1.0f / (float)(numIterations));
		
		FloatImage lightSourceImage = FloatImage::FlipY(dumpImage(this->mResolution, [&] () { runFinalProgram(0.0f, 0.0f, 1.0f, false); }));
		FloatImage photonImage = FloatImage::FlipY(dumpImage(this->mResolution, [&] () { runFinalProgram(0.0f, 1.0f, 0.0f, false); }));
		photonImage *= param;
		FloatImage vplImage = FloatImage::FlipY(dumpImage(this->mResolution, [&] () { runFinalProgram(1.0f, 0.0f, 0.0f, false); }));
		vplImage *= param;

		FloatImage::Save(lightSourceImage + vplImage + photonImage, mDumpCombineFilename);
		FloatImage::Save(lightSourceImage + vplImage, mDumpWeightedVplFilename);
		FloatImage::Save(photonImage, mDumpWeightedPhotonFilename);
	}

	void destroy()
	{
		rt.destroy();
	}

	// PHOTON configuration
	int mNumLightPaths = 0;
	int mNumVplLightPaths = 0;
	int mNumMaxBounce = 0;
	int mNumPhotonsPerLightPath = 0;
	float mRadiusPercentage = 0.f;
	float mPhotonRadius = 0;
	float mPrecomptedPdfMc = 0.0f;

	glm::uvec2 mResolution;
	glm::vec2 mInvResolution;

	unsigned int mRngOffset = 0;
	int mNumMaxIteration = 0;
	EFrame mFrameMode;
	EMis mMisMode;
	float mTimelimitMs;
	float mClampingValue;
	float mClampingStart;

	// number of vpl used in each frame
	bool mJitter;
	bool mUseStat;

	bool mDoDeferredShading = true;
	bool mDoLightTracing = true;
	bool mDoVplSplat = true;
	bool mDoPhotonSplat = true;
	bool mDoLightRender = true;
	bool mDoFinalize = true;

	// Do progressive rendering
	bool mDoProgressive = false;
	bool mDoWriteEveryFrame = false;
	float mAlphaProgressive = 0.7;

	float mTargetRenderingTime = -1;

	std::string mDumpCombineFilename;
	std::string mDumpWeightedPhotonFilename;
	std::string mDumpWeightedVplFilename;
	std::string mStatFilename;

	// VSL parameter
	bool mForceVsl = false;
	float mVslRadiusPercentage = 0.0f;
	float mVslRadius = 0.0f;
	float mVslInvPiRadius2 = 0.0f;

	shared_ptr<RtScene> mScene;

	RealTime rt;
};

const std::map<std::string, RtComPhoton::EFrame> RtComPhoton::EFrameModeStrMap = {
	{"accumulate", RtComPhoton::EFrame::Accumulate},
	{"cleareveryframe", RtComPhoton::EFrame::ClearEveryFrame}
};

const std::map<std::string, RtComPhoton::EMis> RtComPhoton::EMisModeStrMap = {
	{"one", RtComPhoton::EMis::One},
	{"balance", RtComPhoton::EMis::Balance},
	{"max", RtComPhoton::EMis::Max},
	{"power2", RtComPhoton::EMis::Power2},
	{"geometryClamp", RtComPhoton::EMis::GeometryClamp},
	{"geometryBrdfClamp", RtComPhoton::EMis::GeometryBrdfClamp}
};