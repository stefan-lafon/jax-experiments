## JAX experiments
<table>
  <thead>
    <tr>
      <th width="40%"></th>
      <th width="60%">Notebook</th>
    </tr>
  </thead>
  
  <tr>
    <td align="center">
      <img src="assets/cifar.png" width="180px" alt="Quantization Drift">
    </td>
    <td valign="top">
      <h3><a href="notebooks/01_distributed_cnn.ipynb">1. Distributed CIFAR-10 training in JAX and Flax</a></h3>
      <p>Training a simple classifier, with JIT and PMAP for SPMD.</p>
      <p></p><a href="notebooks/01_distributed_cnn.ipynb">Notebook</a></p>
      </td>
  </tr>

  <tr>
    <td align="center">
      <img src="assets/newton.png" width="180px">
    </td>
    <td valign="top">
      <h3><a href="notebooks/02_curvature_and_newton_methods.ipynb">2. Newton's method via AutoDiff</a></h3>
      <p>Using JAX's AutoDiff twice for Newton's method.</p>
      <p><a href="notebooks/02_curvature_and_newton_methods.ipynb">Notebook</a></p>
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="assets/QAT vs PTQ.png" width="180px">
    </td>
    <td valign="top">
      <h3><a href="03_Simple_QAT_in_JAX.ipynb">3. QAT vs PTQ in JAX</a></h3>
      <p>Implementation of QAT and PTQ in JAX and perf comparison.</p>
      <p><a href="notebooks/03_Simple_QAT_in_JAX.ipynb">Notebook</a></p>
    </td>
  </tr>
</table>
