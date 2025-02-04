from qoop.compilation.qcompilation import QuantumCompilation
from qoop.core import ansatz, state
import qiskit

#define a custom ansatz
def custom_ansatz(num_qubits: int) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits)
    thetas = qiskit.circuit.ParameterVector(
        'theta', 2 * num_qubits)
    j = 0
    for i in range(num_qubits):
        qc.rx(thetas[j], i)
        qc.rz(thetas[j + 1], i)
        j += 2
    return qc    

#run the compiler  
num_qubits = 3
compiler = QuantumCompilation(
    u = custom_ansatz(num_qubits),
    vdagger = state.w(num_qubits).inverse()
)

compiler.fast_fit(num_steps=10)
print(compiler.metrics)