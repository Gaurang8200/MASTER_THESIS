# OAK D Franka calibration

Place the calibration files produced with the active OAK D RGB stream here:

1. `output_wp2camera.json`
2. `output_c2f.json`
3. `robot_poses.json`

The calibration images must use the same cropped OAK D RGB geometry as runtime
capture. Runtime keeps RGB undistortion disabled to preserve the image geometry
used by the existing OAK D calibration. The configured calibration resolution is
640 by 400 and the runtime Franka capture is 1280 by 800.

The robot pipeline stops before coordinate conversion when any file is missing.
The older camera calibration remains in `FR_original_calibration` for rollback.
