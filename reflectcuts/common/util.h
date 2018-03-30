#pragma once

#include "reflectcuts.h"
#include "math/math.h"
#include <json/json.hpp>

#include <iostream>
#include <vector>

namespace Util
{
	void SaveObj(const std::string & filename, const std::vector<Vec3> & points);

	inline Vec3 ToVec3(const nlohmann::json & json)
	{
		Vec3 result;
		for (int i = 0;i < 3;i++) { result[i] = json[i]; }
		return result;
	}

	inline Vec4 ToVec4(const nlohmann::json & json)
	{
		Vec4 result;
		for (int i = 0;i < 4;i++) { result[i] = json[i]; }
		return result;
	}
};