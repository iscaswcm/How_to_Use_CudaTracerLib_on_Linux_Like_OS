mitsubacuda --model.name cornellbox/cornellbox.obj \
  --renderer.imageSamples 128 \
  --renderer.pixelSamples 128 \
  --image.fileName CornellBox.png\
  --image.width  800 \
  --image.height 800 \
  --image.write 1 \
  --camera.px 0 \
  --camera.py 1 \
  --camera.pz 3.5 \
  --camera.tx 0 \
  --camera.ty 1 \
  --camera.tz -3.5 \
  --camera.ux 0 \
  --camera.uy 1 \
  --camera.uz 0 \
  --camera.fov 43
#Remarks
#(1) fov = 43, camera can only see CornellBox without any out door objects.
#(2) The more fov than 43,, the more out door objects can be seen.
#(3) The more fov less than 43, the more local objects can be seen.
