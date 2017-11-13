# example_use_CudaTracerLib

First of all, special thanks to CudaTracerLib's author Hannes.

example how to use CudaTracerLib, CudaTracerLib see  https://github.com/hhergeth/CudaTracerLib,

Features:

    (1) A complete example how to use CudaTracerLib library, and with a carlibration in shell.
	
	(2) hostExample will read config from cudatracerlib.ini, if no special command line parameters given.

	(3) Any parameter offered by command line option is prior to the couterpart given by cudatracerlib.ini.


How to use:

$ make

$ ./hostExample --help

Options:

  --help

  --model.path arg (=Data/)

  --model.name arg (=Greeble_big/Greeble_big.obj)

  --general.windowed arg (=0)

  --general.maxCpuThreadCount arg (=0)

  --general.cudaDeviceNumber arg (=0)

  --renderer.type arg (=1)

  --renderer.skip arg (=0)

  --renderer.imageSamples arg (=1)

  --renderer.pixelSamples arg (=1)

  --image.width arg (=1280)

  --image.height arg (=800)

  --image.write arg (=1)

  --image.fileName arg (=image.png) (png or bmp or tga)

  --camera.px arg (=0)

  --camera.py arg (=95)

  --camera.pz arg (=-170)

  --camera.tx arg (=0)

  --camera.ty arg (=95)

  --camera.tz arg (=-169)

  --camera.ux arg (=0)

  --camera.uy arg (=1)

  --camera.uz arg (=0)

  --camera.fov arg (=90)

$ ./hostExample   // This will render cornellbox scene by CudaTracerLib


or just type:

$ ./CornellBox.sh

$ ./Greenbig.sh
