#define OPENGL_VERSION_MAJOR 4
#define OPENGL_VERSION_MINOR 5

#include <iostream>
#include <memory>
#include <fstream>

#include <experimental/filesystem>
namespace fsystem = std::experimental::filesystem;

#if defined(__GNUC__)
#else
#include <conio.h>
#endif

#include "glm/ext.hpp"

// common
#include "common/floatimage/floatimage.h"
#include "common/stopwatch.h"
#include "common/rng.h"
#include "common/util.h"

// shape
#include "shapes/trianglemesh.h"

// math
#include "math/mapping.h"
#include "math/color.h"

// sampler
#include "sampler/independent.h"


// realtime techniques
#include "json/json.hpp"
#include "realtimetechniques/rtcommon.h"
#include "realtimetechniques/rtpt/rtpt2.h"
#include "realtimetechniques/rtcomphoton/rtcomphoton.h"
#include "realtimetechniques/rtcomphoton/rtlvccomphoton.h"

shared_ptr<RtScene> LoadScene(nlohmann::json & json, const std::string& jsonFilename)
{
	shared_ptr<RtScene> rtScene = make_shared<RtScene>();

	if (json["scene"].is_null()) { return nullptr; }

	for (size_t i = 0;i < json["scene"].size();i++)
	{
		std::string objFilename = json["scene"][i];
		fsystem::path p = objFilename;
		if (!p.is_absolute()) {
			// Use the JSON file as the working directory
			fsystem::path pJSON = jsonFilename;
			p = pJSON.parent_path() / fsystem::path(p);
		}
		rtScene->addObject(p.string());
	}

	std::string objFilenameLight = json["arealight"]["obj"];
	fsystem::path p = objFilenameLight;
	if (!p.is_absolute()) {
		// Use the JSON file as the working directory
		fsystem::path pJSON = jsonFilename;
		p = pJSON.parent_path() / fsystem::path(p);
	}
	rtScene->addAreaLight(p.string(), glm::mat4(1.0f), Util::ToVec4(json["arealight"]["intensity"]));
	// all arealight loaded are scaled by Pi

	float aspect = (float) json["resX"] / (float) json["resY"];

	shared_ptr<RtCameraBase> camera;
	if (json.find("camera") != json.end())
	{
		camera = make_shared<RtStableCamera>(json["camera"], aspect);
	}
	else if (json.find("stablecamera") != json.end())
	{
		camera = make_shared<RtStableCamera>(json["stablecamera"], aspect);
	}

	rtScene->setCamera(camera);

	return rtScene;
}

int main(int numArg, const char * args[])
{
	std::string jsonFilename;
	std::ifstream ifs;
	if(numArg > 1)
	{
		jsonFilename = args[1];
	}
	else
	{
		jsonFilename = "../scene/conference/conference_ours.json";
	}
	ifs.open(jsonFilename, std::ios::in);
	// Load JSON file
	nlohmann::json json;
	ifs >> json;

	shared_ptr<RtScene> scene = LoadScene(json, jsonFilename);
	if(!json["pt"].is_null())
	{
		RtPt2 rtpt;
		rtpt.render(scene, glm::uvec2(json["resX"], json["resY"]), json["pt"]);
	}

	if(!json["photonfam"].is_null())
	{
		RtComPhoton rtcomp;
		rtcomp.render(scene, glm::uvec2(json["resX"], json["resY"]), json["photonfam"]);
	}

	if(!json["lvcphotonfam"].is_null())
	{
		RtLvcComPhoton rtcomp2;
		rtcomp2.render(scene, glm::uvec2(json["resX"], json["resY"]), json["lvcphotonfam"]);
	}

	return 0;
}
