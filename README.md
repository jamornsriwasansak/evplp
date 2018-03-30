# Efficient Energy-Compensated VPLs using Photon Splatting

Jamorn Sriwasansak<sup>1</sup>, Adrien Gruson<sup>1,2</sup>, Toshiya Hachisuka<sup>1</sup>

<sup>1</sup>The University of Tokyo <sup>2</sup>JFLI, CNRS, UMI 3527

## Overview
This project contains the source code for the paper "Efficient Energy-Compensated VPLs using Photon Splatting". Along with our technique, we also implemented 
1. Path-Tracing with MIS next-event estimation
2. Instant Radiosity using Virtual Point Light [1]
3. Instant Radiosity using Virtual Spherical Light [2]
4. Image Space Photon Splatting (based on splatting technique in the paper "Hardware-accelerated global illumination by
image space photon mapping" [3])

and their progressive variants
1. Progressive VPL [4]
2. Progressive VSL [5]
3. Progressive Photon Mapping [6,7,8]

## Requirements
1. Microsoft Visual Studio 2015
2. CUDA version 8
3. Optix SDK version >= 4.1.1
4. OpenGL version >= 4.5
5. Assimp 
6. GLEW
7. GLFW
8. GLM
9. nlohmann's json
10. STB

(We had already included the libraries listed in 5 - 10 in the folder "dependencies".)

<img src="readme-resources/main-image.png"/>

## Acknowledgement
Along with the source code we also include 3 scenes (the conference, the living room, and the buddha) that were used in the paper for analysis. 

We thus would like to acknowledge Anat Grynberg and Greg Ward (the conference room), Stanford Computer Graphics Laboratory (happy buddha), blendswap.com artists "cenobi"(the living room).

## License
This rendering framework is released under the MIT license.

## Reference
[1] Alexander Keller. 1997. Instant radiosity. In Proceedings of the 24th annual conference
on Computer graphics and interactive techniques. ACM Press/Addison-Wesley
Publishing Co., 49–56.

[2] Miloš Hašan, Jaroslav Křivánek, Bruce Walter, and Kavita Bala. 2009. Virtual spherical
lights for many-light rendering of glossy scenes. In ACM Transactions on Graphics
(TOG), Vol. 28. ACM, 143.

[3] Morgan McGuire and David Luebke. 2009. Hardware-accelerated global illumination by
image space photon mapping. In Proceedings of the Conference on High Performance
Graphics 2009. ACM, 77–89.

[4] Tomáš Davidovič, Iliyan Georgiev, and Philipp Slusallek. 2012. Progressive lightcuts
for GPU. In ACM SIGGRAPH 2012 Talks. ACM, 1.

[5] Jan Novák, Derek Nowrouzezahrai, Carsten Dachsbacher, and Wojciech Jarosz. 2012a.
Progressive virtual beam lights. In Computer Graphics Forum, Vol. 31. Wiley Online
Library, 1407–1413.

[6] Toshiya Hachisuka, Shinji Ogaki, and Henrik Wann Jensen. 2008. Progressive photon
mapping. ACM Transactions on Graphics (TOG) 27, 5 (2008), 130.

[7] Toshiya Hachisuka and Henrik Wann Jensen. 2009. Stochastic Progressive Photon
Mapping. ACM Trans. Graph. 28, 5, Article 141 (Dec. 2009), 8 pages. https://doi.
org/10.1145/1618452.1618487

[8] Claude Knaus and Matthias Zwicker. 2011. Progressive photon mapping: A probabilistic
approach. ACM Transactions on Graphics (TOG) 30, 3 (2011), 25.