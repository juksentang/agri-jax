"""Site assembly: the layer that turns the files of a site or a reference-model run into the
parameters, forcing and events of an assembled model.

The layers of the package, bottom to top::

    core, iface, forcing      state, process contract, port records, forcing preparation
    io                        file readers and writers (DSSAT, RZWQM2, AmeriFlux, CA-TPA archives)
    processes, models         process modules and the days assembled from them
    sites                     site assembly: reads with io, builds the records of processes

``io`` imports no ``processes`` module and ``processes`` imports no ``io`` module; only this
layer (and the layers above it: calibration, tests, scripts) imports both. The lint rule AJ012
(:mod:`agrijax.core.lint`) enforces it.

* :mod:`agrijax.sites.dssat_inputs`: CERES-Maize parameters and forcing from a DSSAT-CSM run
  (``DSSAT48.INP``, ``*.ECO`` / ``*.SPE``, ``Weather.OUT``, ``SoilWat.OUT``, SPAM dumps);
* :mod:`agrijax.sites.catpa_m3`: the CA-TPA RZWQM2 scenario as the inputs of the RZWQM2 4.6 day;
* :mod:`agrijax.sites.catpa_dssat`: the CA-TPA scenario as a DSSAT-CSM 4.8.6 experiment.

Submodules are imported explicitly (``from agrijax.sites.catpa_m3 import ...``); importing the
package loads none of them.
"""
