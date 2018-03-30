#pragma once

#include "common/reflectcuts.h"
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cassert>

//#include "glm\glm.hpp"

class FloatImage
{
public:
	enum WrapMode
	{
		Repeat,
		Clamp,
		Mirror,
	};

	static FloatImage ComputeSquareErrorHeatImage(const FloatImage & fimage, const FloatImage & refImage, const Float maxError);
	static FloatImage ComputeRelSquareErrorHeatImage(const FloatImage & fimage, const FloatImage & refImage, const Float maxError);

	static Float ComputeMse(const FloatImage & fimage, const FloatImage & refImage);
	static Float ComputeRelMse(const FloatImage & fimage, const FloatImage & refImage);

	static FloatImage FlipY(const FloatImage &fimage);
	static FloatImage Abs(const FloatImage &fimage);
	static FloatImage LoadPFM(const std::string &filepath);
	static void SavePFM(const FloatImage &fimage, const std::string &filepath);
	static FloatImage LoadHDR(const std::string &filepath);
	static void SaveHDR(const FloatImage &image, const std::string &filepath);
	static void SavePNG(const FloatImage &fimage, const std::string &filepath);
	static void Save(const FloatImage &fimage, const std::string &filepath);
	static FloatImage Pow(const FloatImage &image, const float value);
	static FloatImage ResizeBilinear(const FloatImage &image, const glm::uvec2 & size);

	static FloatImage HorizontalBlur(const FloatImage & image, const std::vector<float> & kernel);
	static FloatImage VerticalBlur(const FloatImage & image, const std::vector<float> & kernel);
	static FloatImage GaussianBlur(const FloatImage & image, const float deviation);

	FloatImage() : _mSize(glm::uvec2(0, 0)) {}
	FloatImage(const size_t sizeX, const size_t sizeY, const std::vector<glm::vec3> &data);
	FloatImage(const size_t sizeX, const size_t sizeY);
	FloatImage(const glm::uvec2 &size);

	void setZero();

	inline glm::vec3& colorAt(const size_t x, const size_t y) { assert(x < _mSize.x); assert(y < _mSize.y); return _mData[y * _mSize.x + x]; }
	inline const glm::vec3& colorAt(const size_t x, const size_t y) const { assert(x < _mSize.x); assert(y < _mSize.y); return _mData[y * _mSize.x + x]; }

	//inline glm::vec3& colorAtF(float u, float v) { return this->colorAt(size_t(u * (_mSize.x - 0.0001f)), size_t(v * (_mSize.y - 0.0001f))); }
	//inline const glm::vec3& colorAtF(float u, float v) const { return this->colorAt(size_t(u * (_mSize.x - 0.0001f)), size_t(v * (_mSize.y - 0.0001f))); }

	//inline glm::vec3& colorAtF(const glm::vec2 & uv) { return colorAtF(uv.x, uv.y); }
	//inline const glm::vec3& colorAtF(const glm::vec2 & uv) const { return colorAtF(uv.x, uv.y); }

	glm::vec3 gaussianColorAt(const size_t x, const size_t y, const float deviation);

	glm::vec3 evalTexel(int32_t x, int32_t y) const;
	glm::vec3 evalNearest(const glm::vec2 & uv) const;
	glm::vec3 evalBilinear(const glm::vec2 & uv) const;

	inline const glm::uvec2 getSize() const { return _mSize; }
	inline const size_t getNumPixels() const { return _mSize.x * _mSize.y; }

	WrapMode mWrapS = WrapMode::Repeat;
	WrapMode mWrapT = WrapMode::Repeat;

	inline float* getFloats() { return (float*)(&_mData[0].x); }
	inline const float* getFloats() const { return (float*)(&_mData[0].x); }

	FloatImage operator+(const FloatImage & image) const;

	inline FloatImage & operator+=(const FloatImage & p)
	{
		for (size_t y = 0;y < this->_mSize.y;y++)
			for (size_t x = 0;x < this->_mSize.x;x++)
				this->colorAt(x, y) += p.colorAt(x, y);
		return *this;
	}

	inline FloatImage & operator-=(const FloatImage & p)
	{
	  for (size_t y = 0;y < this->_mSize.y;y++)
		  for (size_t x = 0;x < this->_mSize.x;x++)
			  this->colorAt(x, y) -= p.colorAt(x, y);
	  return *this;
	}

	inline FloatImage & operator/=(const Float v)
	{
		for (size_t y = 0;y < this->_mSize.y;y++)
			for (size_t x = 0;x < this->_mSize.x;x++)
				this->colorAt(x, y) /= v;
		return *this;
	}

	inline FloatImage & operator*=(const Float v)
	{
		for (size_t y = 0;y < this->_mSize.y;y++)
		  for (size_t x = 0;x < this->_mSize.x;x++)
			  this->colorAt(x, y) *= v;
		return *this;
	}

	std::vector<glm::vec3> _mData;
	glm::uvec2 _mSize;
};
