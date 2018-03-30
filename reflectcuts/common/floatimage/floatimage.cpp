#define _CRT_SECURE_NO_DEPRECATE
#include "floatimage.h"

#include <malloc.h>
#include <cassert>
#include <memory>
#include <iostream>
#include <string>
#include <fstream>
#include <xmmintrin.h>
#include <omp.h>
#include <exception>

#include "common/floatimage/rgbe.h"
#include "math/math.h"
#include "math/color.h"
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

FloatImage FloatImage::ComputeSquareErrorHeatImage(const FloatImage & fimage, const FloatImage & refImage, const Float maxError)
{
	assert(fimage.getSize() == refImage.getSize());

	FloatImage out(fimage.getSize());
	
	size_t numRow = fimage.getSize().y;
	size_t numCol = fimage.getSize().x;
	for (size_t y = 0;y < numRow;y++)
	{
		for (size_t x = 0;x < numCol;x++)
		{
			Float meanSquareError = glm::distance2(fimage.colorAt(x, y), refImage.colorAt(x, y));
			out.colorAt(x, y) = glm::vec3(Color::Heat(std::min(meanSquareError / maxError, (Float)1.0)));
		}
	}
	return out;
}

FloatImage FloatImage::ComputeRelSquareErrorHeatImage(const FloatImage & fimage, const FloatImage & refImage, const Float maxError)
{
	assert(fimage.getSize() == refImage.getSize());

	FloatImage out(fimage.getSize());
	
	size_t numRow = fimage.getSize().y;
	size_t numCol = fimage.getSize().x;
	for (size_t y = 0;y < numRow;y++)
	{
		for (size_t x = 0;x < numCol;x++)
		{
			const glm::vec3 & ref = refImage.colorAt(x, y);
			glm::vec3 diff = fimage.colorAt(x, y) - ref;
			Float numerator = glm::dot(diff, diff);
			Float denominator = glm::dot(ref, ref) + (Float)0.001;
			Float relSquareError = numerator / denominator;

			out.colorAt(x, y) = glm::vec3(Color::Heat(std::min(relSquareError / maxError, (Float)1.0)));
		}
	}
	return out;
}

Float FloatImage::ComputeMse(const FloatImage & fimage, const FloatImage & refImage)
{
	assert(fimage.getSize() == refImage.getSize());

	Float result = 0;

	size_t numRow = fimage.getSize().y;
	size_t numCol = fimage.getSize().x;

	Float numPixels = (Float)(numRow * numCol);
	for (size_t y = 0;y < numRow;y++)
	{
		for (size_t x = 0;x < numCol;x++)
		{
			glm::vec3 diff = fimage.colorAt(x, y) - refImage.colorAt(x, y);
			result += glm::dot(diff, diff);
		}
	}

	return result / numPixels;
}

Float FloatImage::ComputeRelMse(const FloatImage & fimage, const FloatImage & refImage)
{
	assert(fimage.getSize() == refImage.getSize());

	Float result = 0;

	size_t numRow = fimage.getSize().y;
	size_t numCol = fimage.getSize().x;

	Float numPixels = (Float)(numRow * numCol);
	for (size_t y = 0;y < numRow;y++)
	{
		for (size_t x = 0;x < numCol;x++)
		{
			const glm::vec3 & ref = refImage.colorAt(x, y);
			glm::vec3 diff = fimage.colorAt(x, y) - ref;
			Float numerator = glm::dot(diff, diff);
			Float denominator = glm::dot(ref, ref) + (Float)0.001;
			Float relSquareError = numerator / denominator;

			result += relSquareError;
		}
	}

	return result / numPixels;

}

FloatImage FloatImage::FlipY(const FloatImage & fimage)
{
	size_t numRow = fimage.getSize().y;
	size_t numCol = fimage.getSize().x;

	FloatImage result(numCol, numRow);
	for (size_t row = 0;row < numRow;row++)
	{
		for (size_t col = 0;col < numCol;col++)
		{
			result._mData[row * numCol + col] = fimage._mData[(numRow - 1 - row) * numCol + col];
		}
	}
	return result;
}

FloatImage FloatImage::Abs(const FloatImage & fimage)
{
	size_t numRow = fimage.getSize().y;
	size_t numCol = fimage.getSize().x;
	FloatImage result(numCol, numRow);
	for (size_t row = 0;row < numRow;row++)
	{
		for (size_t col = 0;col < numCol;col++)
		{
			result._mData[row * numCol + col] = glm::abs(fimage._mData[row * numCol + col]);
		}
	}
	return result;
}

FloatImage FloatImage::LoadPFM(const std::string & filepath)
{
	std::ifstream is(filepath, std::ios::binary | std::ios::in);
	size_t row;
	size_t col;
	std::string pfm;
	std::string minusOne;

	// read pfm header
	is >> pfm;
	is >> col >> row;
	is >> minusOne;
	assert(col > 0);
	assert(row > 0);
	char c = is.get();
	if (c == '\r') c = is.get();

	// compute numpixels
	size_t numPixels = col * row;
	FloatImage result(col, row);
	float* floats = result.getFloats();

	// read data flip row
	for (size_t i = 0;i < row;i++)
	{
		is.read((char*)(&floats[col * (row - i - 1) * 3]), sizeof(float) * col * 3);
	}

	is.close();

	return result;
}

void FloatImage::SavePFM(const FloatImage & fimage, const std::string & filepath)
{
	size_t col = fimage._mSize.x;
	size_t row = fimage._mSize.y;
	size_t numPixels = row * col;

	const float* floats = fimage.getFloats();

	// write PFM header
	std::ofstream os(filepath, std::ios::binary);
	os << "PF" << std::endl;
	os << col << " " << row << std::endl;
	os << "-1" << std::endl;

	// write data flip row
	for (size_t i = 0;i < row;i++)
	{
		os.write((char*)(&floats[col * (row - i - 1) * 3]), sizeof(float) * col * 3);
	}

	os.close();
}

FloatImage FloatImage::LoadHDR(const std::string & filepath)
{
	// open file
	FILE *fp = nullptr;
	fp = fopen(filepath.c_str(), "rb");
	assert(fp != nullptr);
	
	// read col, row
	int col = 0;
	int row = 0;
	rgbe_header_info headerInfo;
	RGBE_ReadHeader(fp, &col, &row, &headerInfo);
	int numPixels = col * row;

	// read into hdrData
	FloatImage result(col, row);
	RGBE_ReadPixels_RLE(fp, result.getFloats(), col, row);
	fclose(fp);

	return result;
}

void FloatImage::SaveHDR(const FloatImage & image, const std::string & filepath)
{
	// open file
	FILE *fp = nullptr;
	fp = fopen(filepath.c_str(), "wb");
	assert(fp != nullptr);

	int numRows = image.getSize().y;
	int numCols = image.getSize().x;
	int numPixels = numCols * numRows;

	// write header and data
	RGBE_WriteHeader(fp, numCols, numRows, NULL);
	RGBE_WritePixels_RLE(fp, image.getFloats(), numCols, numRows);

	fclose(fp);
}

void FloatImage::SavePNG(const FloatImage & fimage, const std::string & filepath)
{
	char * data = new char[fimage.getSize().x * fimage.getSize().y * 3];
	for (size_t row = 0;row < fimage.getSize().y;row++)
	{
		for (size_t col = 0;col < fimage.getSize().x;col++)
		{
			for (size_t i = 0;i < 3;i++)
			{
				Float p = std::pow(fimage.colorAt(col, row)[i], Float(1 / 2.2));
				p = std::min(p * 255.99, 255.0);
				data[(col + row * fimage.getSize().x) * 3 + i] = static_cast<char>(p);
			}
		}
	}
	stbi_write_png(filepath.c_str(), fimage.getSize().x, fimage.getSize().y, 3, data, fimage.getSize().x * 3);
    delete data;
}

void FloatImage::Save(const FloatImage & fimage, const std::string & filepath)
{
	size_t i = filepath.find_last_of('.');
	assert(i > 0 && i < filepath.length() - 1);
	std::string extension = filepath.substr(i + 1);
	if (extension == "pfm")
		SavePFM(fimage, filepath);
	else if (extension == "hdr")
		SaveHDR(fimage, filepath);
	else if (extension == "png")
		SavePNG(fimage, filepath);
	else
		throw std::exception("unsupported file format");
}

FloatImage FloatImage::Pow(const FloatImage & image, const float exponent)
{
	size_t numRows = image.getSize().y;
	size_t numCols = image.getSize().x;
	FloatImage result(image._mSize);

	for (size_t row = 0;row < numRows;row++)
	{
		for (size_t col = 0;col < numCols;col++)
		{
			result._mData[col + row * numCols] = glm::pow(image._mData[col + row * numCols], glm::vec3(exponent));
		}
	}
	return result;
}

FloatImage FloatImage::ResizeBilinear(const FloatImage & image, const glm::uvec2 & size)
{
	FloatImage result(size);
	for (size_t row = 0; row < size.y; row++)
	{
		for (size_t col = 0; col < size.x; col++)
		{
			float u = float(col) / size.x;
			float v = float(row) / size.y;

			result.colorAt(col, row) = image.evalBilinear(glm::vec2(u, v));
		}
	}
	
	return result;
}

FloatImage FloatImage::HorizontalBlur(const FloatImage & image, const std::vector<float> & kernel)
{
	const int kernelRadius = (int)kernel.size() / 2;
	const int rStart = -kernelRadius;
	const int rEnd = rStart + (int)kernel.size();

	if (kernelRadius <= 1) { return image; }

	FloatImage result(image._mSize);

	for (uint32_t y = 0;y < image._mSize.y;y++)
	{
		for (uint32_t x = 0;x < image._mSize.x;x++)
		{
			int start = std::max(int(x) + rStart, 0);
			int end = std::min(int(x) + rEnd, (int)image._mSize.x);

			int n = end - start;
			glm::vec3 sum(0.0f);
			float denominator = 0.0f;
			if (n != 0)
			{
				for (int p = 0, r = start;r < end;r++, p++)
				{
					sum += image.colorAt(r, y) * kernel[p];
					denominator += kernel[p];
				}
			}
			result.colorAt(x, y) = sum / denominator;
		}
	}
	return result;
}

FloatImage FloatImage::VerticalBlur(const FloatImage & image, const std::vector<float> & kernel)
{
	const int kernelRadius = (int)kernel.size() / 2;
	const int rStart = -kernelRadius;
	const int rEnd = rStart + (int)kernel.size();

	if (kernelRadius <= 1) { return image; }

	FloatImage result(image._mSize);

	for (uint32_t y = 0;y < image._mSize.y;y++)
	{
		for (uint32_t x = 0;x < image._mSize.x;x++)
		{
			int start = std::max(int(y) + rStart, 0);
			int end = std::min(int(y) + rEnd, (int)image._mSize.y);

			int n = end - start;
			glm::vec3 sum(0.0f);
			float denominator = 0.0f;
			if (n != 0)
			{
				for (int p = 0, r = start;r < end;r++, p++)
				{
					sum += image.colorAt(x, r) * kernel[p];
					denominator += kernel[p];
				}
			}
			result.colorAt(x, y) = sum / denominator;
		}
	}
	return result;
}

FloatImage FloatImage::GaussianBlur(const FloatImage & image, const float deviation)
{
	// generate 1D gaussian kernel
	int kernelSize = (int)deviation * 3;
	std::vector<float> kernel(kernelSize); // generally 3 x standard deviation is enough

	float term1 = 1.0f / (std::sqrt(2.0f * (float)Math::Pi) * deviation);
	float term2 = 1.0f / (2.0f * deviation * deviation);
	for (int p = 0, i = -kernelSize / 2;i < kernelSize - kernelSize / 2;i++,p++)
	{
		kernel[p] = term1 * std::exp(-i * i * term2);
	}

	FloatImage vertBlur = FloatImage::VerticalBlur(image, kernel);
	return FloatImage::HorizontalBlur(vertBlur, kernel);
}

FloatImage::FloatImage(const size_t sizeX, const size_t sizeY, const std::vector<glm::vec3>& data):
	_mSize(glm::uvec2(sizeX, sizeY)),
	_mData(data),
	mWrapS(FloatImage::WrapMode::Clamp),
	mWrapT(FloatImage::WrapMode::Clamp)
{
	assert(data.size() == sizeX * sizeY);
}

FloatImage::FloatImage(const size_t sizeX, const size_t sizeY) :
	_mSize(glm::uvec2(sizeX, sizeY)),
	_mData(sizeX * sizeY),
	mWrapS(FloatImage::WrapMode::Clamp),
	mWrapT(FloatImage::WrapMode::Clamp)
{
}

FloatImage::FloatImage(const glm::uvec2 & size) :
	FloatImage(size.x, size.y)
{
}

void FloatImage::setZero()
{
	std::memset(&_mData[0].x, 0, sizeof(glm::vec3) * _mSize.x * _mSize.y);
}

glm::vec3 FloatImage::gaussianColorAt(const size_t x, const size_t y, const float deviation)
{
	assert(x < _mSize.x);
	assert(y < _mSize.y);

	int kernelSize = (int)deviation * 3;
	const int kernelRadius = kernelSize / 2;
	const int rStart = -kernelRadius;
	const int rEnd = rStart + kernelSize;

	int startX = std::max((int)x + rStart, 0);
	int endX = std::min((int)x + rEnd, (int)_mSize.x);
	int startY = std::max((int)y + rStart, 0);
	int endY = std::min((int)y + rEnd, (int)_mSize.y);

	float deviation2 = deviation * deviation;
	float term1 = 1.0f / (2.0f * (float)Math::Pi * deviation2);
	float term2 = 0.5f / deviation2;

	glm::vec3 result(0.0f);
	float denominator = 0.0f;
	for (int ry = rStart, py = startY;py < endY;py++, ry++)
	{
		for (int rx = rStart, px = startX;px < endX;px++, rx++)
		{
			float kernelValue = term1 * std::exp(-(float)(rx * rx + ry * ry) * term2);
			result += colorAt((size_t)px, (size_t)py) * kernelValue;
			denominator += kernelValue;
		}
	}

	return result / denominator;
}

glm::vec3 FloatImage::evalTexel(int32_t x, int32_t y) const
{
	const glm::uvec2 & size = _mSize;
	switch (mWrapS)
	{
		case FloatImage::Repeat:
			x = Math::PositiveMod(x, size.x);
			break;
		case FloatImage::Clamp:
			x = Math::Clamp(x, int32_t(0), int32_t(size.x - 1));
			break;
		case FloatImage::Mirror:
			assert(false); /// TODO:: Implement
			break;
		default:
			assert(false);
			break;
	}

	switch (mWrapT)
	{
		case FloatImage::Repeat:
			y = Math::PositiveMod(y, size.y);
			break;
		case FloatImage::Clamp:
			y = Math::Clamp(y, int32_t(0), int32_t(size.y - 1));
			break;
		case FloatImage::Mirror:
			assert(false); /// TODO:: Implement
			break;
		default:
			assert(false);
			break;
	}

	return this->colorAt(x, y);
}

glm::vec3 FloatImage::evalNearest(const glm::vec2 & uv) const
{
	// taken from mitsuba mipmap.h
	const glm::vec2 &size = _mSize;
	return evalTexel(Math::FloorToInt(uv.x * size.x), Math::FloorToInt(uv.y * size.y));
}

glm::vec3 FloatImage::evalBilinear(const glm::vec2 & uv) const
{
	// taken from mitsuba mipmap.h
	const glm::vec2 &size = _mSize;
	float uScaled = uv.x * size.x - 0.5f;
	float vScaled = uv.y * size.y - 0.5f;

	int32_t xPos = Math::FloorToInt(uScaled), yPos = Math::FloorToInt(vScaled);
	float dx1 = uScaled - xPos, dy1 = vScaled - yPos;
	float dx2 = 1.0f - dx1, dy2 = 1.0f - dy1;

	return evalTexel(xPos, yPos) * dx2 * dy2
		+ evalTexel(xPos, yPos + 1) * dx2 * dy1
		+ evalTexel(xPos + 1, yPos) * dx1 * dy2
		+ evalTexel(xPos + 1, yPos + 1) * dx1 * dy1;
}

FloatImage FloatImage::operator+(const FloatImage & image) const
{
	assert(image._mSize == this->_mSize);
	FloatImage result(this->_mSize);
	for (size_t y = 0;y < this->_mSize.y;y++)
		for (size_t x = 0;x < this->_mSize.x;x++)
			result.colorAt(x, y) = this->colorAt(x, y) + image.colorAt(x, y);
	return result;
}