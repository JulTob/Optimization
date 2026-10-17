# 📚 Optimization Reference Library

This directory contains reference materials for convex geometry and optimization theory.

## 📖 Available Resources

### Handbook of Convex Geometry (2 Volume Set)

**Editors:** Peter M. Gruber & J.M. Wills  
**Publisher:** North-Holland (Elsevier)  
**Year:** 1993  
**ISBN:** 0-444-89598-1 (Complete Set)

---

#### 📕 Volume A: Classical Convexity & Combinatorial Aspects

**File:** `Handbook_Convex_Geometry_Part_A_Gruber.pdf`  
**Size:** 46.6 MB  
**Pages:** ~700

**Table of Contents:**
- **Part 1: Classical Convexity**
  - 1.1. Characterizations of convex sets (P. Mani-Levitska)
  - 1.2. Mixed volumes (J.R. Sangwine-Yager)
  - 1.3. The standard isoperimetric theorem (G. Talenti)
  - 1.4. Stability of geometric inequalities (H. Groemer)
  - 1.5. Selected affine isoperimetric inequalities (E. Lutwak)
  - 1.6. Extremum problems for convex discs and polyhedra (A. Florian)
  - 1.7. Rigidity (R. Connelly)
  - 1.8. Convex surfaces, curvature and surface area measures (R. Schneider)
  - 1.9. The space of convex bodies (P.M. Gruber)
  - 1.10. Aspects of approximation of convex bodies (P.M. Gruber)
  - 1.11. Special convex bodies (E. Heil and H. Martini)

- **Part 2: Combinatorial Aspects of Convexity**
  - 2.1. Helly, Radon, and Carathéodory type theorems (J. Eckhoff)
  - 2.2. Problems in discrete and combinatorial geometry (P. Schmitt)
  - 2.3. Combinatorial aspects of convex polytopes (M.M. Bayer and C.W. Lee)
  - 2.4. Polyhedral manifolds (U. Brehm and J.M. Wills)
  - 2.5. Oriented matroids (J. Bokowski)

---

#### 📕 Volume B: Geometric Applications & Advanced Topics

**File:** `Handbook_Convex_Geometry_Volume_B_Gruber_Wills.pdf`  
**Size:** 14.0 MB  
**Pages:** 770

**Key Topics for Optimization:**

##### Support Functions & Gauge Functions (Page 365)
- **Indicator function:** $I(x|U) = \begin{cases} 0 & \text{if } x \in U, \\ +\infty & \text{if } x \notin U \end{cases}$
- **Gauge function:** $G(x|U) = \inf\{\lambda \geqslant 0 \mid x \in \lambda U\}$
- **Support function:** $S(\ell, U) = \sup\{\langle \ell, x \rangle \mid x \in U\}$
- **Distance function:** $D(x|U) = \inf\{\|x - y\| \mid y \in U\}$

##### Conjugate Convex Functions
- Young's inequality applications
- Fenchel's work on support functions and gauge functions
- Connections to optimization duality

##### Minkowski Geometry (Multiple Sections)
- Minkowski addition of convex bodies
- Reduction theory (Lagrange, Seeber, Gauss, Minkowski)
- LLL-basis and Lovász reduction
- Minkowski's theorem on convex bodies of constant width

##### Spherical Harmonics & Fourier Analysis (Page 554)
- Harmonic expansion of support functions
- Mean width and Steiner point calculations
- $L_2$ approximation of convex bodies
- Parseval's equation applications

##### Packing & Covering Problems
- Gap size and overlap size definitions
- Packing with convex sets
- Covering with translates of convex bodies

##### Mixed Volumes & Quermassintegrals
- Geometric inequalities
- Minkowski formulas
- Rotation integrals

---

## 🎯 Most Relevant Sections for Optimization

### For Convex Optimization Theory:
1. **Support Functions** (Vol B, p. 365) - Fundamental for duality
2. **Conjugate Functions** (Vol B, p. 365) - Basis for Lagrange duality
3. **Minkowski Addition** (Vol B, p. 249, 641) - Subdifferential calculus
4. **Approximation Theory** (Vol A, Ch. 1.10) - Polyhedral approximations

### For Computational Geometry:
1. **Helly's Theorem** (Vol A, Ch. 2.1) - Intersection properties
2. **Radon's Theorem** (Vol A, Ch. 2.1) - Partitioning convex sets
3. **Carathéodory's Theorem** (Vol A, Ch. 2.1) - Convex hull representation
4. **Polytope Theory** (Vol A, Ch. 2.3) - LP feasible regions

### For Advanced Topics:
1. **Isoperimetric Inequalities** (Vol A, Ch. 1.3, 1.5) - Optimization on manifolds
2. **Rigidity Theory** (Vol A, Ch. 1.7) - Structural optimization
3. **Stability of Inequalities** (Vol A, Ch. 1.4) - Robust optimization

---

## 🔗 External Resources

- **Z-Library:** [https://z-lib.org](https://z-lib.org) - Source of these handbooks
- **Springer Handbook of Convex Geometry:** Alternative reference
- **Boyd & Vandenberghe, "Convex Optimization":** [https://web.stanford.edu/~boyd/cvxbook/](https://web.stanford.edu/~boyd/cvxbook/)

---

## 📝 Citation Information

```bibtex
@book{gruber1993handbook,
  title={Handbook of Convex Geometry},
  editor={Gruber, Peter M. and Wills, J{"o}rg M.},
  volume={A and B},
  year={1993},
  publisher={North-Holland},
  address={Amsterdam},
  isbn={0-444-89598-1 (set), 0-444-89596-5 (Vol A), 0-444-89597-3 (Vol B)}
}
```

---

## 💡 Usage Notes

These handbooks are **comprehensive reference works** containing:
- **Theoretical foundations** of convex geometry
- **Historical development** of the field
- **Advanced topics** for research
- **Extensive bibliographies** for further reading

**For your Optimization Wiki:** Focus on Volume B for practical optimization applications, especially the sections on support functions, conjugate functions, and Minkowski geometry.

---

## 🎨 Visualization Resources

See the Colab notebooks in the repository for interactive visualizations of concepts from these handbooks.
