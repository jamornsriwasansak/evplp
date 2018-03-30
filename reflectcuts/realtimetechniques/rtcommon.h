#pragma once

#include <iostream>
#include <vector>
#include <map>

#include "common/reflectcuts.h"
#include "common/util.h"
#include "math/aabb.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <vector_types.h>

#include "opengl/buffer.h"

struct RtTexture
{
	inline static float FromSRGBComponent(float value)
	{
		if (value <= (float) 0.04045) return value * (float) (1.0 / 12.92);
		return std::pow((value + (float) 0.055) * (float) (1.0 / 1.055), (float) 2.4);
	}

	static shared_ptr<RtTexture> LoadRtTexture(const std::string & filedir, const aiMaterial & aiMat, const aiTextureType & textureKey, const char * colorKey, unsigned int type, unsigned int index)
	{
		stbi_set_flip_vertically_on_load(1);
		static std::map<const std::string, shared_ptr<RtTexture>> gTexturesMap;
		aiString texturePath;
		aiColor3D reflectance;	// kd

		if (aiMat.Get(AI_MATKEY_TEXTURE(textureKey, 0), texturePath) == aiReturn_SUCCESS)
		{
			std::string textureFilepath = filedir + texturePath.C_Str();
			std::map<const std::string, shared_ptr<RtTexture>>::iterator texturesMapIterator = gTexturesMap.find(textureFilepath);

			if (texturesMapIterator == gTexturesMap.end())
			{
				shared_ptr<RtTexture> result = make_shared<RtTexture>(textureFilepath, 1.0f);
				gTexturesMap[textureFilepath] = result;
				return result;
			}
			else
			{
				return texturesMapIterator->second;
			}
		}
		else if (aiMat.Get(colorKey, type, index, reflectance) == aiReturn_SUCCESS)
		{
			if (textureKey == aiTextureType::aiTextureType_SHININESS)
			{
				// fix shininess bug introduced by assimp (they said "to match what most renderers do")
				shared_ptr<RtTexture> tex = make_shared<RtTexture>(reflectance, 1.0f);
				tex->mData[0] /= 4.0f;
				tex->mData[1] /= 4.0f;
				tex->mData[2] /= 4.0f;
				tex->mData[3] /= 4.0f;
				return tex;
			}

			//return make_shared<RtTexture>(reflectance, true);
			return make_shared<RtTexture>(reflectance, 1.0f);
			//return make_shared<RtTexture>(reflectance, 2.2f);
		}

		throw std::exception(); // "Loading Texture Error"
	}

	RtTexture()
	{
		mIsExistGl = false;
		mIsExistOptix = false;
	}

	RtTexture(const float r, const float g, const float b, const float gamma):
		mSize(glm::uvec2(1, 1)),
		mIsExistGl(false),
		mIsExistOptix(false)
	{
		mData.resize(4);
		mData[0] = std::pow(r, gamma);
		mData[1] = std::pow(g, gamma);
		mData[2] = std::pow(b, gamma);
		mData[3] = 0.f;
	}

	RtTexture(const float r, const float g, const float b, const bool useSrgb):
		mSize(glm::uvec2(1, 1)),
		mIsExistGl(false),
		mIsExistOptix(false)
	{
		mData.resize(4);
		if (useSrgb)
		{
			mData[0] = FromSRGBComponent(r);
			mData[1] = FromSRGBComponent(g);
			mData[2] = FromSRGBComponent(b);
		}
		else
		{
			mData[0] = r;
			mData[1] = g;
			mData[2] = b;
		}
		mData[3] = 0.f;
	}

	RtTexture(const FloatImage & image):
		mSize(image.getSize()),
		mIsExistGl(false),
		mIsExistOptix(false)
	{
		mData.resize(mSize.x * mSize.y * 4);

		for (size_t i = 0;i < mSize.x * mSize.y;i++)
		{
			for (size_t j = 0;j < 3;j++)
			{
				mData[i * 4 + j] = image._mData[i][j];
			}
		}
	}

	RtTexture(const aiColor3D & color, const bool useSrgb):
		RtTexture(color.r, color.g, color.b, useSrgb)
	{
	}

	RtTexture(const aiColor3D & color, const float gamma):
		RtTexture(color.r, color.g, color.b, gamma)
	{
	}

	RtTexture(const std::string & filepath, const float gamma):
		mIsExistGl(false),
		mIsExistOptix(false)
	{
		int width = 0, height = 0, channel = 0;
		stbi_uc * data = stbi_load(filepath.c_str(), &width, &height, &channel, 3);

		assert(data != nullptr);
		assert(width > 0);
		assert(height > 0);
		assert(channel == 1 || channel == 3 || channel == 4);

		size_t dataSize = width * height;
		mSize = glm::uvec2(width, height);
		mData.resize(dataSize * 4);

		if (channel == 1)
		{
			for (size_t i = 0;i < dataSize;i++)
			{
				for (size_t j = 0;j < 3;j++)
				{
					// convert from rgb to float accurately
					mData[i * 4 + j] = std::pow(static_cast<float>(data[i * 3]) / 255.0f, gamma);
				}
				mData[i * 4 + 3] = 0.f;
			}
		}
		if (channel == 3)
		{
			for (size_t i = 0;i < dataSize;i++)
			{
				for (size_t j = 0;j < 3;j++)
				{
					// convert from rgb to float accurately
					mData[i * 4 + j] = std::pow(static_cast<float>(data[i * 3 + j]) / 255.0f, gamma);
				}
				mData[i * 4 + 3] = 0.f;
			}
		}
		else if (channel == 4)
		{
			for (size_t i = 0;i < dataSize;i++)
			{
				for (size_t j = 0;j < 3;j++)
				{
					// convert from rgb to float accurately
					mData[i * 4 + j] = std::pow(static_cast<float>(data[i * 4 + j]) / 255.0f, gamma);
				}
				mData[i * 4 + 3] = 0.f;
			}
		}


		stbi_image_free(data);
	}

	void createOpenglTexture()
	{
		// must create in 4 channels due to optix limitation
		if (!mIsExistGl)
		{
			glGenTextures(1, &mGlHandle);
			glBindTexture(GL_TEXTURE_2D, mGlHandle);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mSize.x, mSize.y, 0, GL_RGBA, GL_FLOAT, mData.data());

			/*if (mUseSrgb)
			{
				glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, mSize.x, mSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, mData.data());
			}
			else
			{
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, mSize.x, mSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, mData.data());
			}*/

			mIsExistGl = true;
		}
	}

	void createOptixTexture(optix::Context context)
	{
		mOptixTexture.mSampler = context->createTextureSampler();
		mOptixTexture.mSampler->setWrapMode(0, RT_WRAP_REPEAT);
		mOptixTexture.mSampler->setWrapMode(1, RT_WRAP_REPEAT);
		mOptixTexture.mSampler->setWrapMode(2, RT_WRAP_REPEAT);
		mOptixTexture.mSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
		mOptixTexture.mSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
		mOptixTexture.mSampler->setMaxAnisotropy(1.0f);
		mOptixTexture.mSampler->setMipLevelCount(1);
		mOptixTexture.mSampler->setArraySize(1);

		mOptixTexture.mBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, mSize.x, mSize.y);
		float * bufferData = static_cast<float*>(mOptixTexture.mBuffer->map());
		for (size_t i = 0;i < mSize.x * mSize.y * 4;i++)
		{
			bufferData[i] = mData[i];
		}
		mOptixTexture.mBuffer->unmap();

		mOptixTexture.mSampler->setBuffer(0, 0, mOptixTexture.mBuffer);
		mOptixTexture.mSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	}

	void createOptixTextureSamplerFromOpenglTexture(optix::Context context)
	{
		if (!mIsExistOptix)
		{
			assert(mIsExistGl);
			try
			{
				this->mOptixTexture.mSampler = context->createTextureSamplerFromGLImage(mGlHandle, RTgltarget::RT_TARGET_GL_TEXTURE_2D);
				mIsExistOptix = true;
			}
			catch (const optix::Exception & e)
			{
				std::cout << e.what() << std::endl;
//				_getch();
			}
		}
	}

	glm::uvec2					mSize;
	std::vector<float>			mData;
	struct OptixTexture
	{
		optix::Buffer			mBuffer;
		optix::TextureSampler	mSampler;
	} mOptixTexture;
	GLuint						mGlHandle;
	bool						mIsExistGl;
	bool						mIsExistOptix;
	bool						mUseSrgb;
};

struct RtMaterial
{
	RtMaterial()
	{}

	void createOpenglTextures()
	{
		mLambertReflectance->createOpenglTexture();
		mPhongReflectance->createOpenglTexture();
		mPhongExponent->createOpenglTexture();
	}

	void createOptixTextures(optix::Context ctx)
	{
		mLambertReflectance->createOptixTexture(ctx);
		mPhongReflectance->createOptixTexture(ctx);
		mPhongExponent->createOptixTexture(ctx);
	}

	void createOptixTexturesFromOpenglTextures(optix::Context ctx)
	{
		mLambertReflectance->createOptixTextureSamplerFromOpenglTexture(ctx);
		mPhongReflectance->createOptixTextureSamplerFromOpenglTexture(ctx);
		mPhongExponent->createOptixTextureSamplerFromOpenglTexture(ctx);
	}

	shared_ptr<RtTexture>	mLambertReflectance;
	shared_ptr<RtTexture>	mPhongReflectance;
	shared_ptr<RtTexture>	mPhongExponent;
	glm::vec4				mLightIntensity;
};

struct RtMesh
{
	RtMesh(): mNumVertices(0)
	{}

	void applyTransform(const glm::mat4 & transformMatrix)
	{
		glm::vec3 * vertices = reinterpret_cast<glm::vec3*>(mVertices.data());
		glm::vec3 * normals = reinterpret_cast<glm::vec3*>(mNormals.data());

		glm::mat4 transformMatrix_Normal = glm::inverseTranspose(transformMatrix);

		for (size_t i = 0;i < mNumVertices;i++)
		{
			vertices[i] = transformMatrix * glm::vec4(vertices[i], 1.0f);
			normals[i] = transformMatrix_Normal * glm::vec4(normals[i], 0.0f);
			normals[i] = glm::normalize(normals[i]);
		}
	}

	void createOptixMeshBuffer(optix::Context ctx)
	{
		try
		{
			mOptix.mIndices = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, this->mNumTriangles);
			mOptix.mVertices = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, this->mNumVertices);
			mOptix.mNormals = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, this->mNumVertices);
			mOptix.mTexCoords = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, this->mNumVertices);

			// map buffers
			int32_t* indices = reinterpret_cast<int32_t*>(mOptix.mIndices->map());
			float* vertices = reinterpret_cast<float*>(mOptix.mVertices->map());
			float* normals = reinterpret_cast<float*>(mOptix.mNormals->map());
			float* texcoords = reinterpret_cast<float*>(mOptix.mTexCoords->map());

			std::memcpy(indices, mTriIndices.data(), 3 * this->mNumTriangles * sizeof(int32_t));
			std::memcpy(vertices, mVertices.data(), 3 * this->mNumVertices * sizeof(float));
			std::memcpy(normals, mNormals.data(), 3 * this->mNumVertices * sizeof(float));
			std::memcpy(texcoords, mTexCoords.data(), 2 * this->mNumVertices * sizeof(float));

			// unmap buffers
			mOptix.mIndices->unmap();
			mOptix.mVertices->unmap();
			mOptix.mNormals->unmap();
			mOptix.mTexCoords->unmap();
		}
		catch (const optix::Exception & e)
		{
			std::cout << e.what() << std::endl;
//			_getch();
		}
	}

	void createOpenglBuffer()
	{
		assert(mTexCoords.size() / 2 == mVertices.size() / 3);

		mGl.mVerticesBuffer = make_shared<OpenglBuffer>();
		mGl.mNormalsBuffer = make_shared<OpenglBuffer>();
		mGl.mTexCoordsBuffer = make_shared<OpenglBuffer>();
		mGl.mIndicesBuffer = make_shared<OpenglBuffer>();

		glNamedBufferData(mGl.mVerticesBuffer->mHandle, sizeof(float) * mNumVertices * 3, &(mVertices[0]), GL_STATIC_DRAW);
		glNamedBufferData(mGl.mNormalsBuffer->mHandle, sizeof(float) * mNumVertices * 3, &(mNormals[0]), GL_STATIC_DRAW);
		glNamedBufferData(mGl.mTexCoordsBuffer->mHandle, sizeof(float) * mNumVertices * 2, &(mTexCoords[0]), GL_STATIC_DRAW);
		glNamedBufferData(mGl.mIndicesBuffer->mHandle, sizeof(uint32_t) * mNumTriangles * 3, &(mTriIndices[0]), GL_STATIC_DRAW);
	}

	// doesn't work
	void createOptixMeshBufferFromOpenglBuffer(optix::Context context)
	{
		size_t elementSize;

		// setup vertices
		mOptix.mVertices = context->createBufferFromGLBO(RT_BUFFER_INPUT, mGl.mVerticesBuffer->mHandle);
		mOptix.mVertices->setFormat(RTformat::RT_FORMAT_FLOAT3);
		mOptix.mVertices->setSize(mNumVertices);

		// setup normals
		mOptix.mNormals = context->createBufferFromGLBO(RT_BUFFER_INPUT, mGl.mNormalsBuffer->mHandle);
		mOptix.mNormals->setFormat(RTformat::RT_FORMAT_FLOAT3);
		mOptix.mNormals->setSize(mNumVertices);

		// setup texcoords
		mOptix.mTexCoords = context->createBufferFromGLBO(RT_BUFFER_INPUT, mGl.mTexCoordsBuffer->mHandle);
		mOptix.mTexCoords->setFormat(RTformat::RT_FORMAT_FLOAT2);
		mOptix.mTexCoords->setSize(mNumVertices);

		// setup indices
		mOptix.mIndices = context->createBufferFromGLBO(RT_BUFFER_INPUT, mGl.mIndicesBuffer->mHandle);
		mOptix.mIndices->setFormat(RTformat::RT_FORMAT_UNSIGNED_INT3);
		mOptix.mIndices->setSize(mNumTriangles);
	}

	void createOptixGeometry(optix::Context context, optix::Program meshIntersectProgram, optix::Program boundingBoxProgram, optix::Material optixMaterial, const RtMaterial & rtMaterial)
	{
		try
		{
			mOptix.mGeometry = context->createGeometry();
			mOptix.mGeometry->setPrimitiveCount(mNumTriangles);
			mOptix.mGeometry->setIntersectionProgram(meshIntersectProgram);
			mOptix.mGeometry->setBoundingBoxProgram(boundingBoxProgram);

			mOptix.mGeometry["indexBuffer"]->setBuffer(mOptix.mIndices);
			mOptix.mGeometry["vertexBuffer"]->setBuffer(mOptix.mVertices);
			mOptix.mGeometry["normalBuffer"]->setBuffer(mOptix.mNormals);
			mOptix.mGeometry["texcoordBuffer"]->setBuffer(mOptix.mTexCoords);

			mOptix.mGeometryInstance = context->createGeometryInstance(mOptix.mGeometry, &optixMaterial, &optixMaterial + 1);
			mOptix.mGeometryInstance["lambertReflectanceTexture"]->setTextureSampler(rtMaterial.mLambertReflectance->mOptixTexture.mSampler);
			mOptix.mGeometryInstance["phongReflectanceTexture"]->setTextureSampler(rtMaterial.mPhongReflectance->mOptixTexture.mSampler);
			mOptix.mGeometryInstance["phongExponentTexture"]->setTextureSampler(rtMaterial.mPhongExponent->mOptixTexture.mSampler);
			mOptix.mGeometryInstance["lightIntensity"]->set4fv(&(rtMaterial.mLightIntensity[0]));
		}
		catch (const optix::Exception & e)
		{
			std::cout << e.what() << std::endl;
			throw std::exception();
		}
	}

	Aabb computeBbox()
	{
		Aabb result;
		glm::vec3 * vertices = reinterpret_cast<glm::vec3*>(mVertices.data());
		for (size_t i = 0;i < mNumVertices;i++) { result = Aabb::Union(result, vertices[i]); }
		return result;
	}

	float recomputeArea() const
	{

		float sumArea = 0.f;
		for (size_t i = 0; i < mNumTriangles; i++)
		{
			// load indices
			glm::vec3 vertices[3];
			for (size_t j = 0; j < 3; j++)
			{
				unsigned int index = this->mTriIndices[i * 3 + j];
				vertices[j] = glm::vec3(this->mVertices[index * 3], this->mVertices[index * 3 + 1], this->mVertices[index * 3 + 2]);
			}
			float area = Triangle::ComputeArea(vertices[0], vertices[1], vertices[2]);
			sumArea += area;
		}

		return sumArea;
	}


	int32_t						mNumVertices;
	int32_t						mNumTriangles;
	int32_t						mMatIndex;

	std::vector<float>			mVertices;
	std::vector<float>			mNormals;
	std::vector<float>			mTexCoords;
	std::vector<int32_t>		mTriIndices;

	struct OptixMeshBuffer
	{
		optix::GeometryInstance mGeometryInstance;
		optix::Geometry			mGeometry;
		optix::Buffer			mIndices;
		optix::Buffer			mVertices;
		optix::Buffer			mNormals;
		optix::Buffer			mTexCoords;
	} mOptix;

	struct OpenglMeshBuffers
	{
		shared_ptr<OpenglBuffer> mVerticesBuffer;
		shared_ptr<OpenglBuffer> mNormalsBuffer;
		shared_ptr<OpenglBuffer> mIndicesBuffer;
		shared_ptr<OpenglBuffer> mTexCoordsBuffer;
	} mGl;
};

struct RtAreaLight
{
	RtAreaLight()
	{
	}

	RtAreaLight(shared_ptr<RtMesh> mesh, const glm::vec4 & lightIntensity, const glm::vec4 & mPrecomputedLightIntensity):
		mMesh(mesh),
		mLightIntensity(lightIntensity),
		mPrecomputedLightIntensity(mPrecomputedLightIntensity)
	{
	}

	void createOptixCdf(optix::Context ctx)
	{
		const size_t numTriangles = mMesh->mNumTriangles;
		this->mOptixCdfBuffer = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, numTriangles);

		float* cdf = reinterpret_cast<float*>(mOptixCdfBuffer->map());

		float sumArea = 0.f;
		for (size_t i = 0;i < numTriangles;i++)
		{
			// load indices
			glm::vec3 vertices[3];
			for (size_t j = 0;j < 3;j++)
			{
				unsigned int index = this->mMesh->mTriIndices[i * 3 + j];
				vertices[j] = glm::vec3(this->mMesh->mVertices[index * 3], this->mMesh->mVertices[index * 3 + 1], this->mMesh->mVertices[index * 3 + 2]);
			}
			float area = Triangle::ComputeArea(vertices[0], vertices[1], vertices[2]);
			sumArea += area;
			cdf[i] = sumArea;
		}

		// normalize cdf
		for (size_t i = 0;i < numTriangles;i++)
		{
			cdf[i] /= sumArea;
		}

		mMeshArea = sumArea;
		mOptixCdfBuffer->unmap();
	}

	float				mMeshArea;
	shared_ptr<RtMesh>	mMesh;
	glm::vec4           mLightIntensity;
	glm::vec4			mPrecomputedLightIntensity;
	optix::Buffer		mOptixCdfBuffer;
};

struct RtCameraBase
{
	virtual glm::vec3 getOrigin() const = 0;
	virtual glm::mat4 computeVpMatrix(float nearDist = 0.1f, float farDist = 100.0f) const = 0;
};

struct RtStableCamera : RtCameraBase
{
	RtStableCamera(const nlohmann::json & json, float aspectRatio)
	{
		float fovy = 0;
		if (json.find("fovy") != json.end())
		{
			float fovyDegree = json["fovy"];
			fovy = glm::radians(fovyDegree);
		}
		else if (json.find("fovx") != json.end())
		{
			float fovxDegree = json["fovx"];
			fovy = 2.0f * std::atan2(std::tan(glm::radians(fovxDegree) * 0.5f), aspectRatio);
		}
		else
		{
			throw std::exception(); // "forgot fov"
		}

		mOrigin = Util::ToVec3(json["origin"]);
		mLookAt = Util::ToVec3(json["direction"]);
		mUp = Util::ToVec3(json["up"]);
		mFovy = fovy;
		mAspectRatio = aspectRatio;
	}

	RtStableCamera(const glm::vec3 & origin, const glm::vec3 & lookAt, const glm::vec3 & up, const float fovy, const float aspectRatio):
		mOrigin(origin),
		mLookAt(lookAt),
		mUp(up),
		mFovy(fovy),
		mAspectRatio(aspectRatio)
	{}

	glm::vec3 getOrigin() const override
	{
		return mOrigin;
	}

	glm::mat4 computeVpMatrix(float nearDist = 0.1f, float farDist = 100.0f) const override
	{
		glm::mat4 viewMatrix = glm::lookAt(mOrigin, mLookAt, mUp);
		glm::mat4 projectionMatrix = glm::perspective(mFovy, mAspectRatio, nearDist, farDist);
		return projectionMatrix * viewMatrix;
	}

	glm::vec3			mOrigin;
	glm::vec3			mLookAt;
	glm::vec3			mUp;
	float				mFovy;
	float				mAspectRatio;
};

struct RtAnimationCamera : RtCameraBase
{
	glm::vec3 getOrigin() const override
	{
		return glm::lerp(mOriginStart, mOriginEnd, currentTime / allTime);
	}

	glm::mat4 computeVpMatrix(float nearDist = 0.1f, float farDist = 100.0f) const override
	{
		glm::vec3 origin = getOrigin();
		glm::vec3 up = glm::lerp(mUpStart, mUpEnd, currentTime / allTime);
		glm::vec3 lookAt = glm::lerp(mLookAtStart, mLookAtEnd, currentTime / allTime);
		glm::mat4 viewMatrix = glm::lookAt(origin, lookAt, up);
		glm::mat4 projectionMatrix = glm::perspective(mFovy, mAspectRatio, nearDist, farDist);
		return projectionMatrix * viewMatrix;
	}

	float               currentTime;
	float               allTime;              // amount of time spend from origin to end in millisecond
	float               timeStep;             // size of time step eg. 0.1 ms per frame
	
	glm::vec3           mOriginStart;
	glm::vec3           mOriginEnd;
	glm::vec3           mLookAtStart;
	glm::vec3           mLookAtEnd;
	glm::vec3           mUpStart;
	glm::vec3           mUpEnd;
	float               mFovy;
	float               mAspectRatio;
};

struct RtScene
{
	static bool GetTextureFilepath(std::string * filepath, const std::string & filedir, const aiMaterial & aiMat, const aiTextureType & textureKey, const char * colorKey, unsigned int type, unsigned int index)
	{
		aiString texturePath;
		if (aiMat.Get(AI_MATKEY_TEXTURE(textureKey, 0), texturePath) == aiReturn_SUCCESS)
		{
			*filepath = filedir + texturePath.C_Str();
			return true;
		}
		return false;
	}

	void addObject(const std::string & filepath,
		const glm::mat4 & modelMatrix = glm::mat4(),
		const glm::vec4 & lightIntensity = glm::vec4(0.0f),
		bool overrideMaterial = false,
		shared_ptr<RtMaterial> defaultMat = make_shared<RtMaterial>())
	{
		const unsigned int aiProcesses = aiProcess_Triangulate
			| aiProcess_GenSmoothNormals
			| aiProcess_JoinIdenticalVertices
			| aiProcessPreset_TargetRealtime_Fast;

		// load aiScene
		Assimp::Importer importer;
		const aiScene * scene = importer.ReadFile(filepath.c_str(), aiProcesses);
		if (scene == nullptr) {
			std::cerr << "Impossible to load the scene: " << filepath << "\n";
			assert(false);
		}

		const size_t matOffset = this->mMaterials.size();

		// populate rtmesh result vector
		for (size_t iMesh = 0;iMesh < scene->mNumMeshes;iMesh++)
		{
			shared_ptr<RtMesh> mesh = make_shared<RtMesh>();

			const auto aiSceneMesh = scene->mMeshes[iMesh];
			const size_t numVertices = aiSceneMesh->mNumVertices;

			assert(aiSceneMesh->HasNormals());

			for (size_t iVert = 0;iVert < numVertices;iVert++)
			{
				// position
				{
					const float x = aiSceneMesh->mVertices[iVert][0];
					const float y = aiSceneMesh->mVertices[iVert][1];
					const float z = aiSceneMesh->mVertices[iVert][2];

					mesh->mVertices.push_back(x);
					mesh->mVertices.push_back(y);
					mesh->mVertices.push_back(z);
				}

				// normal
				{
					mesh->mNormals.push_back(aiSceneMesh->mNormals[iVert][0]);
					mesh->mNormals.push_back(aiSceneMesh->mNormals[iVert][1]);
					mesh->mNormals.push_back(aiSceneMesh->mNormals[iVert][2]);
				}

				// tex coords
				if (aiSceneMesh->HasTextureCoords(0))
				{
					mesh->mTexCoords.push_back(aiSceneMesh->mTextureCoords[0][iVert][0]);
					mesh->mTexCoords.push_back(aiSceneMesh->mTextureCoords[0][iVert][1]);
				}
				else
				{
					mesh->mTexCoords.push_back(0.0f);
					mesh->mTexCoords.push_back(0.0f);
				}
			}

			const size_t numTriangles = aiSceneMesh->mNumFaces;
			for (size_t iIdx = 0;iIdx < numTriangles;iIdx++)
			{
				mesh->mTriIndices.push_back(aiSceneMesh->mFaces[iIdx].mIndices[0]);
				mesh->mTriIndices.push_back(aiSceneMesh->mFaces[iIdx].mIndices[1]);
				mesh->mTriIndices.push_back(aiSceneMesh->mFaces[iIdx].mIndices[2]);
			}

			const size_t matIndex = aiSceneMesh->mMaterialIndex;
			mesh->mNumVertices		= numVertices;
			mesh->mNumTriangles		= numTriangles;
			if (overrideMaterial)
			{
				mesh->mMatIndex = matOffset;
			}
			else
			{ 
				mesh->mMatIndex = matOffset + matIndex;
			}

			mesh->applyTransform(modelMatrix);

			this->mMeshes.push_back(mesh);
		}

		std::string filedir = filepath.substr(0, filepath.find_last_of("/\\")) + "\\";

		if (overrideMaterial)
		{
			mMaterials.push_back(defaultMat);
		}
		else
		{
			// populate rtmaterial result vector
			const size_t numMaterials = scene->mNumMaterials;
			for (size_t iMat = 0;iMat < numMaterials;iMat++)
			{
				// note: assimp always generate material index = 0 as DefaultMaterial automatically
				const auto aiMat = scene->mMaterials[iMat];
				shared_ptr<RtMaterial> mat = make_shared<RtMaterial>();

				mat->mLambertReflectance = RtTexture::LoadRtTexture(filedir, *aiMat, aiTextureType_DIFFUSE, AI_MATKEY_COLOR_DIFFUSE);
				mat->mPhongReflectance = RtTexture::LoadRtTexture(filedir, *aiMat, aiTextureType_SPECULAR, AI_MATKEY_COLOR_SPECULAR);
				mat->mPhongExponent = RtTexture::LoadRtTexture(filedir, *aiMat, aiTextureType_SHININESS, AI_MATKEY_SHININESS);
				mat->mLightIntensity = lightIntensity;

				mMaterials.push_back(mat);
			}
		}
	}

	float totalArea() const {
		float sumArea = 0.f;
		for (auto m : mMeshes) {
			sumArea += m->recomputeArea();
		}

		//sumArea += mArealight->mMeshArea;

		return sumArea;
	}

	// support only one area light source! -> merge two separated arealight as one arealight works as well
	bool isAlreadyHaveLightSource = false;
	void addAreaLight(const std::string & filepath, const glm::mat4 & modelMatrix = glm::mat4(), const glm::vec4 & lightIntensity = glm::vec4(0.0f))
	{
		assert(!isAlreadyHaveLightSource);

		size_t beforeSize = mMeshes.size();

		isAlreadyHaveLightSource = true;

		glm::vec4 precomputedLightIntensity = lightIntensity;

		for (size_t i = 0;i < 3;i++) { precomputedLightIntensity[i] = lightIntensity[i] * Math::Pi; }

		shared_ptr<RtMaterial> mat = make_shared<RtMaterial>();
		mat->mLambertReflectance = make_shared<RtTexture>(0.f, 0.f, 0.f, 1.f);
		mat->mPhongExponent = make_shared<RtTexture>(0.f, 0.f, 0.f, 1.f);
		mat->mPhongReflectance = make_shared<RtTexture>(0.f, 0.f, 0.f, 1.f);
		mat->mLightIntensity = precomputedLightIntensity;

		this->addObject(filepath, modelMatrix, precomputedLightIntensity, true, mat);

		size_t afterSize = mMeshes.size();

		// we can't have > 1 mesh per light source
		assert(afterSize - beforeSize == 1);
		
		this->mArealight = make_shared<RtAreaLight>(this->mMeshes.back(), lightIntensity, precomputedLightIntensity);
	}

	void setCamera(shared_ptr<RtCameraBase> camera)
	{
		this->mCamera = camera;
	}

	float findBoundingSphereRadius()
	{
		Aabb bbox;
		for (shared_ptr<RtMesh> mMesh : mMeshes)
		{
			bbox = Aabb::Union(bbox, mMesh->computeBbox());
		}
		float diameter = std::sqrt(Aabb::DiagonalLength2(bbox));
		return diameter / 2.0f;
	}

	shared_ptr<RtAreaLight>					mArealight;
	std::vector<shared_ptr<RtMesh>>			mMeshes;
	std::vector<shared_ptr<RtMaterial>>		mMaterials;
	shared_ptr<RtCameraBase>				mCamera;
};
