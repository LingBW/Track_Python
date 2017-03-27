# Track_Python
If you put a particle in the ocean, you might wonder where it gonna go.  “Particle Tracking” could give you some answers based on model FVCOM and ROMS. We have four cases for tracking.

Case 1: Drifter tracking, If you deploy a drifter, just input your drifter-ID and the days you want to forecast, “particle tracking” will show you the forecast trajectory, click [sample 1](Samples of Animation/Option-1-drifter_track.gif) to show you the sample.

Case 2: Coordinate tracking. You are allowed specify  one or two coordinates of degree for tracking, the days of forecast are needed also. The result looks like [this](Samples of Animation/Option-3.gif). We also provide option to add points between two points given using function ‘points_between()’. More detail see the codes.

Case 3: Click tracking. In this case, a simulated map will show np, you click on the map to add points for tracking. Days are needed ,too. [The result](Samples of Animation/Option-3.gif) same with Case 2.

Case 4: Box tracking. You specify one coordinate of points. Function ‘extend_units’ will extend to a box points in the center of the point you given. Click see [the sample](Samples of Animation/Option4.gif).

All of results can be shown in two types, image(.png) and animation(.gif). There are options,  for example boundary-line, streamline, wind, can be added to the results. [Here](Samples of Animation/streamline.gif) is a streamline shown on the result.
