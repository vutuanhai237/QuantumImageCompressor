import numpy as np
import pennylane as qml
import pennylane.numpy as nps
import qiskit
from .constant import tau
from .utils import state_to_qc
def divide_image(img, k):
    # Add padding if needed
    if int(np.log2(k**2)) != np.log2(k**2):
        raise ValueError("k must be a power of 2")
    h, w = img.shape
    if h % k != 0 or w % k != 0:
        new_h = h + (k - h % k) if h % k != 0 else h
        new_w = w + (k - w % k) if w % k != 0 else w
        padded_img = np.zeros((new_h, new_w), dtype=img.dtype)
        padded_img[:h, :w] = img
        img = padded_img

    blocks = []
    scales = []
    h, w = img.shape
    for i in range(0, h, k):
        for j in range(0, w, k):
            block = img[i:i+k, j:j+k].flatten()
            norm = np.linalg.norm(block)
            if norm != 0:
                block = block / norm
            blocks.append(block)
            scales.append(norm)
    return blocks, scales

dev = qml.device("default.qubit")

def circuit_curry(num_qubits, num_layers):
    @qml.qnode(dev)
    def circuit(thetas):
        k = 0
        for _ in range(num_layers):
            for i in range(0, num_qubits - 1):
                qml.CNOT(wires = [i, i+1])
            qml.CNOT(wires = [num_qubits - 1, 0])
            for i in range(0, num_qubits):
                qml.RX(thetas[k], wires = i)
                qml.RY(thetas[k+1], wires = i)
                qml.RZ(thetas[k+2], wires = i)
                k += 3
        return qml.state()
    return circuit
    
def cost_curry(state, circuit):
    def cost_func(thetas):
        psi = nps.expand_dims((state), axis = 1)
        phi = nps.expand_dims(circuit(thetas), axis = 1)
        p0 = (nps.real(nps.dot(nps.conjugate(psi).T, phi))**2)[0][0]
        return 1 - p0
    return cost_func


def compilation(state, thetas, num_layers, steps = 100, opt = qml.AdamOptimizer(stepsize = 0.1)):
    # Thetas must be nps array
    
    num_qubits = int(np.log2(state.shape[0]))
    
    circuit = circuit_curry(num_qubits, num_layers)
    cost_func = cost_curry(state, circuit)
    grad_func = qml.grad(cost_func)

    costs = []
    thetass = []
    if thetas is None:
        thetas = nps.array(np.ones(3*num_layers*num_qubits))
    for i in range(steps):
        thetas, prev_cost = opt.step_and_cost(cost_func, thetas, grad_fn = grad_func)
        costs.append(prev_cost)
        thetass.append(thetas)
        if prev_cost < tau:
            print(f"Achieved error threshold at step {i}")
            break
    print(i)
    return costs, thetass, i
