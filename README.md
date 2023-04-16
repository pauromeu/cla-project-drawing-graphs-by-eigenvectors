# Drawing Graphs by Eigenvector

This project focused on spectral graph drawing methods, which construct the layout of a graph using eigenvectors of certain matrices. One possible approach is the force-directed strategy [1], which defines an energy function, with the minimum determining a drawing that is optimal in a certain sense. Another approach using the generalized eigenvectors of the graph Laplacian was proposed in [2].

a) State Theorem 1 from [2] and prove it in your own words.

b) Explain in your own words Section 3.1 of [2] about the derivation of (13) and show that it can be solved by solving an eigenvalue problem.

c) Prove in your own words that the generalized eigenvectors of (L,D) defined in section 4 of [2] are generalized eigenvectors of (A,D) in reverse order. Also, prove that the generalized eigevectors of (A,D) coincide with eigenvectors of D−1A when D is symmetric positive definite.

d) Implement the algorithm in Figure 3 from [2] and test it on simple small graphs. The graph can be obtained from https://houseofgraphs.org/

Hint: Use the graph and plot function in Matlab for visualizing the graph.

e) Generate larger graphs by sampling the following random graph models and test them
with your implementation.
– Barabási–Albert model, see https://barabasi.com/f/622.pdf.
– Stochastic block model, see section 2.1 of [3].

## References

[1] Giuseppe Di Battista, Peter Eades, Roberto Tamassia, and Ioannis G Tollis. Graph drawing:
algorithms for the visualization of graphs. Prentice Hall PTR, 1998.

[2] Y. Koren. Drawing graphs by eigenvectors: theory and practice. Comput. Math. Appl.,
49(11-12):1867–1888, 2005.

[3] Jing Lei and Alessandro Rinaldo. Consistency of spectral clustering in stochastic block
models. Ann. Statist., 43(1):215–237, 2015.