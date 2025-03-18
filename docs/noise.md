## Digital Noise

In the description of closed quantum systems, the complete quantum state is a pure quantum state represented by a state vector $|\psi \rangle $.

However, this description is not sufficient to describe open quantum systems. When the system interacts with its surrounding environment, it transitions into a mixed state where quantum information is no longer entirely contained in a single state vector but is distributed probabilistically.

To address these more general cases, we consider a probabilistic combination $p_i$ of possible pure states $|\psi_i \rangle$. Thus, the system is described by a density matrix $\rho$ defined as follows:

$$
\rho = \sum_i p_i |\psi_i\rangle \langle \psi_i|
$$

The transformations of the density operator of an open quantum system interacting with its noisy environment are represented by the super-operator $S: \rho \rightarrow S(\rho)$, often referred to as a quantum channel.
Quantum channels, due to the conservation of the probability distribution, must be CPTP (Completely Positive and Trace Preserving). Any CPTP super-operator can be written in the following form:

$$
S(\rho) = \sum_i K_i \rho K^{\dagger}_i
$$

Where $K_i$ are Kraus operators satisfying the closure property $\sum_i K_i K^{\dagger}_i = \mathbb{I}$. As noise is the result of system interactions with its environment, it is therefore possible to simulate noisy quantum circuit with noise type gates.

Thus, `horqrux` implements a large selection of single qubit noise gates such as:

- The bit flip channel defined as: $\textbf{BitFlip}(\rho) =(1-p) \rho + p X \rho X^{\dagger}$
- The phase flip channel defined as: $\textbf{PhaseFlip}(\rho) = (1-p) \rho + p Z \rho Z^{\dagger}$
- The depolarizing channel defined as: $\textbf{Depolarizing}(\rho) = (1-p) \rho + \frac{p}{3} (X \rho X^{\dagger} + Y \rho Y^{\dagger} + Z \rho Z^{\dagger})$
- The pauli channel defined as: $\textbf{PauliChannel}(\rho) = (1-p_x-p_y-p_z) \rho
            + p_x X \rho X^{\dagger}
            + p_y Y \rho Y^{\dagger}
            + p_z Z \rho Z^{\dagger}$
- The amplitude damping channel defined as: $\textbf{AmplitudeDamping}(\rho) =  K_0 \rho K_0^{\dagger} + K_1 \rho K_1^{\dagger}$
    with:
    $\begin{equation*}
    K_{0} \ =\begin{pmatrix}
    1 & 0\\
    0 & \sqrt{1-\ \gamma }
    \end{pmatrix} ,\ K_{1} \ =\begin{pmatrix}
    0 & \sqrt{\ \gamma }\\
    0 & 0
    \end{pmatrix}
    \end{equation*}$
- The phase damping channel defined as: $\textbf{PhaseDamping}(\rho) = K_0 \rho K_0^{\dagger} + K_1 \rho K_1^{\dagger}$
    with:
    $\begin{equation*}
    K_{0} \ =\begin{pmatrix}
    1 & 0\\
    0 & \sqrt{1-\ \gamma }
    \end{pmatrix}, \ K_{1} \ =\begin{pmatrix}
    0 & 0\\
    0 & \sqrt{\ \gamma }
    \end{pmatrix}
    \end{equation*}$
* The generalize amplitude damping channel is defined as: $\textbf{GeneralizedAmplitudeDamping}(\rho) = K_0 \rho K_0^{\dagger} + K_1 \rho K_1^{\dagger} + K_2 \rho K_2^{\dagger} + K_3 \rho K_3^{\dagger}$
    with:
$\begin{cases}
K_{0} \ =\sqrt{p} \ \begin{pmatrix}
1 & 0\\
0 & \sqrt{1-\ \gamma }
\end{pmatrix} ,\ K_{1} \ =\sqrt{p} \ \begin{pmatrix}
0 & 0\\
0 & \sqrt{\ \gamma }
\end{pmatrix} \\
K_{2} \ =\sqrt{1\ -p} \ \begin{pmatrix}
\sqrt{1-\ \gamma } & 0\\
0 & 1
\end{pmatrix} ,\ K_{3} \ =\sqrt{1-p} \ \begin{pmatrix}
0 & 0\\
\sqrt{\ \gamma } & 0
\end{pmatrix}
\end{cases}$

Noise protocols can be added to gates by instantiating `DigitalNoiseInstance` providing the `DigitalNoiseType` and the `error_probability` (either float or tuple of float):

```python exec="on" source="material-block" html="1"
from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType

noise_prob = 0.3
AmpD = DigitalNoiseInstance(DigitalNoiseType.AMPLITUDE_DAMPING, error_probability=noise_prob)

```

Then a gate can be instantiated by providing a tuple of `DigitalNoiseInstance` instances. Let’s show this through the simulation of a realistic $X$ gate.

For instance, an $X$ gate flips the state of the qubit: $X|0\rangle = |1\rangle$. In practice, it's common for the target qubit to stay in its original state after applying $X$ due to the interactions between it and its environment. The possibility of failure can be represented by the `BitFlip` subclass of `DigitalNoiseInstance`, which flips the state again after the application of the $X$ gate, returning it to its original state with a probability `1 - gate_fidelity`.

```python exec="on" source="material-block"
from horqrux import X, QuantumCircuit
from horqrux.api import sample
from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType
from horqrux.utils.operator_utils import density_mat, product_state

noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.1),)
ops = QuantumCircuit(1, [X(0)])
noisy_ops = QuantumCircuit(1, [X(0, noise=noise)])
state = product_state("0")

noiseless_samples = sample(state, ops)
noisy_samples = sample(density_mat(state), noisy_ops)
print(f"Noiseless samples: {noiseless_samples}") # markdown-exec: hide
print(f"Noiseless samples: {noisy_samples}") # markdown-exec: hide
```
