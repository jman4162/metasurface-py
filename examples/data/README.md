# Reference data for paper reproductions

## faenzi2019_reference_metrics.csv

Reference values for:

> M. Faenzi, G. Minatti, D. González-Ovejero, F. Caminita, E. Martini,
> C. Della Giovampaola, and S. Maci, "Metasurface Antennas: New Models,
> Applications and Realizations," *Scientific Reports* 9:10178 (2019).
> DOI: [10.1038/s41598-019-46522-z](https://doi.org/10.1038/s41598-019-46522-z)

The paper is published under a CC-BY-4.0 license. All values in the CSV
are quoted from the paper's text and figure captions (measured gain,
bandwidth edges, efficiencies, reactance pairs), not extracted from the
plotted curves.

Full measured pattern curves (Fig. 5a, Fig. 6) are not included: they
would have to be digitized from the published raster figures. To add
them, export from WebPlotDigitizer as CSV with columns
`theta_deg,directivity_dbi` and name the files
`faenzi2019_fig5a_measured.csv` / `faenzi2019_fig6_measured.csv`;
`examples/09_reproduce_faenzi2019.py` overlays them automatically if
present.
