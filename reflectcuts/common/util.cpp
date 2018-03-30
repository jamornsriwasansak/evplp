#include "util.h"

#include <fstream>

void Util::SaveObj(const std::string & filename, const std::vector<Vec3>& points)
{
	std::ofstream fileWriter;
	fileWriter.open(filename, std::ios::binary);

	for (const Vec3 &point : points)
	{
		fileWriter << "v " << point.x << " " << point.y << " " << point.z << std::endl;
	}

	fileWriter.close();
}
