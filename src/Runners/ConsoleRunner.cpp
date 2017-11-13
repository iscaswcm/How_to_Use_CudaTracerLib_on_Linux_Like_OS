﻿#include <Engine/Core.h>
#include <Engine/DynamicScene.h>
#include <SceneTypes/Node.h>
#include <Engine/Material.h>
#include <Base/Buffer.h>
#include <SceneTypes/Light.h>
#include <Engine/Image.h>
#include <Integrators/ProgressivePhotonMapping/VolEstimators/Beam.h>
/////////////////////////////////////////////////////////////////////
#include <Integrators/PathTracer.h>
#include <Integrators/Bidirectional/BDPT.h>
#include <Integrators/PhotonTracer.h>
#include <Integrators/PseudoRealtime/WavefrontPathTracer.h>
#include <Integrators/ProgressivePhotonMapping/PPPMTracer.h>
#include <Integrators/ProgressivePhotonMapping/VolEstimators/Beam.h>
/////////////////////////////////////////////////////////////////////
#include <Kernel/ImagePipeline/ImagePipeline.h>
#include <Kernel/Tracer.h>
#include <iostream>

#include "Utils/Log.h"
#include "Utils/Settings.h"
#include "Utils/StringUtils.h"
#include "Utils/SysUtils.h"
#include "Kernel/Tracer.h"
#include "Runners/ConsoleRunner.h"
#include "Core/App.h"
using namespace std;
using namespace CudaTracerLib;
using namespace std::chrono;//The elements in this header deal with time
//const std::string data_path = "Data/";
class SimpleFileManager : public IFileManager
{
public:
	SimpleFileManager(const std::string _data_path):data_path(_data_path)
	{
	}
	virtual std::string getCompiledMeshPath(const std::string& name)
	{
		return data_path + "Compiled/" + name;
	}
	virtual std::string getTexturePath(const std::string& name)
	{
		return data_path + name;
	}
	virtual std::string getCompiledTexturePath(const std::string& name)
	{
		return data_path + "Compiled/" + name;
	}
	const std::string data_path;
};

void SetupScene(const std::string modelPath, const std::string modelName, Sensor& camera, 
		const Vec3f position, const Vec3f target, const Vec3f up, DynamicScene& scene)
{
	auto ground = scene.CreateNode(modelPath + modelName);
	//AABB: [(-87.7185, 3.5413, -87.1465) - (90.2189, 179.174, 92.0413)]
	//(0, 95, -170)(0, 95, -169)(0, 1, 0)
	//camera.SetToWorld(Vec3f(0,95,-170), Vec3f(0,95,-169), Vec3f(0,1,0));
	camera.SetToWorld(position, target, up);
}

int ConsoleRunner::run()
{
	//extern Coord position;
	Settings& settings = App::getSettings();
	Log& log = App::getLog();

	samplesPerSecondAverage.setAlpha(0.05f);

    printf("\033[32;49;2m");
	Timer totalElapsedTimer;
	const std::string modelPath = settings.model.path;
	const std::string modelName = settings.model.name;
	char outputFileName[256];
	log.logInfo("modelPath = %s, modelName = %s", modelPath.c_str(), modelName.c_str());
	const int width = settings.image.width;
	const int height = settings.image.height;
	const float fov = settings.camera.fov;
	const Vec3f position(settings.camera.px, settings.camera.py, settings.camera.pz);
	const Vec3f target(settings.camera.tx, settings.camera.ty, settings.camera.tz);
	const Vec3f up(settings.camera.ux, settings.camera.uy, settings.camera.uz);
	const int imageSamples = settings.renderer.imageSamples;
	const int pixelSamples = settings.renderer.pixelSamples;
	log.logInfo("width = %u, height = %u, fov = %f, imageSamples = %u", width, height, fov, imageSamples);
	log.logInfo("filename = %s, imageSamples = %u, pixelSamples = %u", 
			settings.image.fileName.c_str(), settings.renderer.imageSamples, pixelSamples);
	//format output image name
	std::string sceneName = modelName;
	std::string findstr = "/";
	sceneName.replace(sceneName.find(findstr), findstr.length(),"_");
	sprintf(outputFileName, "Scene_%s_imageSamples_%d_pixelSamples_%d_%s.png",
			sceneName.c_str(), imageSamples, pixelSamples);
	//format output image name
	InitializeCuda4Tracer(modelPath);
	SimpleFileManager fManager(modelPath);;
	Sensor camera = CreateAggregate<Sensor>(PerspectiveSensor(width, height, fov));
	DynamicScene scene(&camera, SceneInitData::CreateForScene(10, 10, 1000), &fManager);
	SetupScene(modelPath, modelName, camera, position, target, up, scene);
	log.logInfo("Scene %d bounding box %s", modelName, scene.getSceneBox());
	Image outImage(width, height);

	//WavefrontPathTracer tracer;//PhotonTracer, PathTracer, ...
	//WavefrontVCM tracer;//PhotonTracer, PathTracer, ...
	//PmmTracer tracer;//PhotonTracer, PathTracer, ...
	//PhotonTracer tracer;//PhotonTracer, PathTracer, ...
	//PPPMTracer tracer;//PhotonTracer, PathTracer, ...
	BDPT tracer;//BDPTPathTracer, ...
	tracer.Resize(width, height);
	tracer.InitializeScene(&scene);
	scene.UpdateScene();
	
	interrupted = false;
	
	SysUtils::setConsoleTextColor(ConsoleTextColor::WHITE_ON_BLACK);
	uint64_t totalSamples = uint64_t(width) * uint64_t(height) * uint64_t(imageSamples);
	std::atomic<uint32_t> totalSampleCount;
	totalSampleCount = 0;
	int currentpass = 0;
	std::cout << tfm::format("\nRendering started (size: %dx%d, pixels: %s, image samples: %d, pixel samples: %d, total samples: %d)\n\n",
		width,
		height,
		StringUtils::humanizeNumber(double(width * height)),
		imageSamples,
		pixelSamples,
		StringUtils::humanizeNumber(double(totalSamples)));

	Timer renderingElapsedTimer;
	renderingElapsedTimer.setAveragingAlpha(0.05f);
	renderingElapsedTimer.setTargetValue(float(totalSamples));

	std::atomic<bool> renderThreadFinished(false);
	std::exception_ptr renderThreadException = nullptr;
    //lambuda function, symbol '&' means all the variables above are referenced.
	auto renderThreadFunction = [&]()
	{
		try
		{
			if(!settings.renderer.skip)
			{
				for (currentpass = 0; currentpass < imageSamples; currentpass++)
				{
					tracer.DoPass(&outImage, !currentpass);
				}
			}
		}
		catch (...)
		{
			renderThreadException = std::current_exception();
		}

		renderThreadFinished = true;
	};
	//main thread asynchronously runs thread entry function -- renderThreadFunction.
	//look up renderer.render function for how to it work.
	std::thread renderThread(renderThreadFunction);
    //below while loop judges render job wether ends or not & update related parameters.
	while (!renderThreadFinished)
	{
		totalSampleCount = width * height * (currentpass+1);
		renderingElapsedTimer.updateCurrentValue(float(totalSampleCount));

		auto elapsed = renderingElapsedTimer.getElapsed();
		auto remaining = renderingElapsedTimer.getRemaining();

		if(elapsed.totalMilliseconds > 0)
		{
			samplesPerSecondAverage.addMeasurement(float(totalSampleCount) / 
					(float(elapsed.totalMilliseconds) / 1000.0f));
		}

		printProgress(renderingElapsedTimer.getPercentage(), 
				elapsed, remaining, pixelSamples, currentpass);
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
	}
    //main thread comes here & ends.
	renderThread.join();

	if(renderThreadException != nullptr)
		std::rethrow_exception(renderThreadException);

	applyImagePipeline(tracer, outImage, CreateAggregate<Filter>(BoxFilter()));
	outImage.WriteDisplayImage(outputFileName);
	outImage.Free();
	DeInitializeCuda4Tracer();
	renderingElapsedTimer.updateCurrentValue(float(totalSampleCount));

	auto elapsed = renderingElapsedTimer.getElapsed();
	auto remaining = renderingElapsedTimer.getRemaining();

	printProgress(renderingElapsedTimer.getPercentage(), elapsed, remaining, pixelSamples, currentpass);

	float totalSamplesPerSecond = 0.0f;

	if(elapsed.totalMilliseconds > 0)
	{
		totalSamplesPerSecond = float(totalSampleCount) / (float(elapsed.totalMilliseconds) / 1000.0f);
	}
	std::cout << tfm::format("\n\nRendering %s (time: %s, samples/s: %s)\n\n",
		interrupted ? "interrupted" : "finished",
		elapsed.getString(true),
		StringUtils::humanizeNumber(totalSamplesPerSecond));
	SysUtils::setConsoleTextColor(ConsoleTextColor::DEFAULT);
	log.logInfo("Writing final image %s ......Done", outputFileName);
	log.logInfo("Total elapsed time: %s", totalElapsedTimer.getElapsed().getString(true));
    printf("\033[0m");

	return 0;
}

void ConsoleRunner::interrupt()
{
	interrupted = true;
}

void ConsoleRunner::printProgress(float percentage, const TimerData& elapsed, 
		const TimerData& remaining, uint32_t pixelSamples, int currentpass) {
	uint32_t barCount = uint32_t(percentage + 0.5f)/ 4;
    tfm::printf("[");

	for (uint32_t i = 0; i < barCount; ++i)
        tfm::printf(">");

	if(barCount < 25)
	{
        tfm::printf(">");

		for (uint32_t i = 0; i < (24 - barCount); ++i)
            tfm::printf(" ");
	}

    tfm::printf("] ");
    tfm::printf("%.2f %% | ", percentage);
    tfm::printf("Elapsed: %s | ", elapsed.getString());
    tfm::printf("Remaining: %s | ", remaining.getString());
	tfm::printf("Samples/s: %s | ", StringUtils::humanizeNumber(samplesPerSecondAverage.getAverage()));
	tfm::printf("Pixel samples: %d | ", pixelSamples);
	tfm::printf("Current pass: %d", currentpass);
    tfm::printf("          \r");
}