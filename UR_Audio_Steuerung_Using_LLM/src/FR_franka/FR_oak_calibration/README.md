# OAK D Franka calibration

Place the calibration files produced with the active OAK D RGB stream here:

1. `output_wp2camera.json`
2. `output_c2f.json`
3. `robot_poses.json`

The calibration images must come from the OAK D left mono camera on `CAM_B`.
The configured left mono calibration resolution is 640 by 400. Runtime projects
the selected RGB point and its aligned depth into this left mono pixel grid
before calling `pixel2robot`.

The robot pipeline stops before coordinate conversion when any file is missing.
The older camera calibration remains in `FR_original_calibration` for rollback.
