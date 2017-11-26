mitsubacuda  --model.name CornellBox_ori/CornellBox-Glossy.obj \
  --renderer.imageSamples 4 \
  --renderer.pixelSamples 4 \
  --image.fileName glossy.png \
  --image.width  1280 \
  --image.height 1000 \
  --image.write 1 \
  --camera.px -0.0159 \
  --camera.py 0.795 \
  --camera.pz 3.65 \
  \
  --camera.tx -0.0159 \
  --camera.ty 0.795 \
  --camera.tz 0 \
  \
  --camera.ux 0 \
  --camera.uy 1 \
  --camera.uz 0 \
  --camera.fov 43
#Remarks
#(1) fov = 43, camera can only see CornellBox without any out door objects.
#(2) The more fov than 43,, the more out door objects can be seen.
#(3) The more fov less than 43, the more local objects can be seen.
