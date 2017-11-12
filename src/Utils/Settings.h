// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.
#ifndef _SETTING_H_
#define _SETTING_H_

class Settings
{
public:

	bool load(int argc, char** argv);
	struct Model
	{
		std::string path;
		std::string name;
	} model;

	struct General
	{
		bool windowed;
		uint32_t maxCpuThreadCount;
		uint32_t cudaDeviceNumber;
	} general;

	struct Renderer
	{
		uint32_t type;
		bool skip;
		uint32_t imageSamples;
		uint32_t pixelSamples;
	} renderer;

	struct Image
	{
		uint32_t width;
		uint32_t height;
		bool write;
		std::string fileName;
	} image;

	struct Camera
	{
		float px;
		float py;
		float pz;
		float tx;
		float ty;
		float tz;
		float ux;
		float uy;
		float uz;
		float fov;
	} camera;
};
#endif
