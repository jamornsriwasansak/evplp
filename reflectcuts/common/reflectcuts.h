#pragma once

#include <iostream>
#include <memory>

// REFLECTCUTS - offline rendering framework
// (was modified for interactive offline rendering)

// common things
using std::cout;
using std::endl;
using std::unique_ptr;
using std::shared_ptr;
using std::weak_ptr;
using std::make_unique;
using std::make_shared;
using std::static_pointer_cast;
using std::dynamic_pointer_cast;

// forward declaration of important base classes inside common sorted from a to z
class Accel;
class FloatImage;
class Camera;
class Intersection;
class Light;
class Logger;
class Material;
class Rng32;
class Rng64;
class Scene;
class Shape;
class StopWatch;
class Sampler;
class Technique;
class Texture;

#define USE_DETERMINISTIC_RESULT
//#define USE_SINGLE_THREAD 

// math things
#define GLM_FORCE_SIZE_T_LENGTH
#include <glm/glm.hpp>
#include <glm/ext.hpp>

//#define USE_DOUBLE_PRECISION
#ifdef USE_DOUBLE_PRECISION
#define Float double
using Vec4 = glm::dvec4;
using Vec3 = glm::dvec3;
using Vec2 = glm::dvec2;
using Mat4 = glm::dmat4;
using Mat3 = glm::dmat3;
using Mat2 = glm::dmat2;
using Rng = Rng64;
#else
#define Float float
using Vec4 = glm::vec4;
using Vec3 = glm::vec3;
using Vec2 = glm::vec2;
using Mat4 = glm::mat4;
using Mat3 = glm::mat3;
using Mat2 = glm::mat2;
using Rng = Rng32;
#endif

using Ivec2 = glm::ivec2;
using Ivec3 = glm::ivec3;
using Ivec4 = glm::ivec4;
using Uvec2 = glm::uvec2;
using Uvec3 = glm::uvec3;
using Uvec4 = glm::uvec4;