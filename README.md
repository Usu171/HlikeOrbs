# Hydrogenic Orbitals Visualized

![Hydrogenic Orbitals Visualized](assets/1.png)

![Dirac Orbitals 1s~4f](assets/dirac_f.webp)
Dirac orbital 1s2p3d4f density, current, and four components ($\psi_1$, $\psi_2$ (top) $\psi_3$, $\psi_4$ (bottom)), with k ranging from -4 (bottom) to 3 (top) and m from -7/2 (left) to 7/2 (right) (excluding m=0).  
Isosurface level at 0.1×max value. All plots are scaled, isosurface sizes are not directly comparable.  
Generated using `fig_grid.py`

![Dirac Orbitals 1s~6h](assets/dirac_h.png)
1s-6h version  
k: -6 - 5  
m: -11/2 - 11/2

> New image using new [gl-mesh3d](https://github.com/Usu171/gl-mesh3d)  
> It supports cyclic color mapping, avoid phase coloring issues  
> [New plotly.js](https://github.com/Usu171/data/blob/main/plotly-251118-modify.min.js) built based on this library

## Dependencies

- plotly
- numpy
- scipy (>=1.15.0)
- scikit-image

## Usage

see examples
