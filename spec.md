App: a fast, portable, modular sky simulator for visualisation and demos, based on real science

Aims: Generate realistic broad-band optical/IR images of a simulated sky containing galaxies and stars. The entire sky should be seeded and deterministic, so the same position gets the same structure for a given seed. It should use either real or analytical expressions for the cosmic web/large scale structure, then seed galaxies. Galaxies should have mock SFHs, and morphology should reflect SFH with some scatter. Galaxy morhpolgoy should primarily be Sersic, elliptical, or bulge+disc models for now, following redshift dependent mass-metallicity relationship, mass-size relationship etc. Number density/GSMF should reflect true observed distributions. Galaxies will be assigned photometry based on some nearest-neighbor/emulator approach given properties. 

On top of this, a model for disc/halo/brown dwarf stars and their angular dependence should be included, and stars of realistic type and luminosity should be seeded on top of the galaxies.

Room should be left to add other layers here - it should be modular. 

Then there will be a basic layer of telescope simulation - PSF modelling, background noise, using a relevant pixel sclae etc).

On a technical level, performance and portability should be emphasized over absolute accuracy. This may be included as a mod, or in a web app with some backend etc, so it should be most self-contained. Ideally it will be as rapid as possible in generating images and make as many optimizations as possible. 
