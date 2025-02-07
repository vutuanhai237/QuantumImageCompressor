import numpy as np
import qiskit
import pennylane as qml

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

def divide_image_into_blocks(img, k):
	# Add padding if needed
	pad_h = (k - img.shape[0] % k) % k
	pad_w = (k - img.shape[1] % k) % k
	img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
	
	# Divide image into blocks
	blocks = []
	for i in range(0, img_padded.shape[0], k):
		row_blocks = []
		for j in range(0, img_padded.shape[1], k):
			row_blocks.append(img_padded[i:i+k, j:j+k])
		blocks.append(row_blocks)
	
	return blocks

def create_random_indices(n, m):
    pairs = [(i, j) for i in range(n) for j in range(m)]
    np.random.shuffle(pairs)
    return pairs


def state_to_qc(state: np.ndarray) -> qiskit.QuantumCircuit:
    num_qubits = int(np.log2(state.shape[0]))
    qc = qiskit.QuantumCircuit(num_qubits)
    qc.prepare_state(state)
    qcx = qiskit.transpile(qc, basis_gates=['h','s','cx','u','rx','ry','rz'], optimization_level=3).inverse()
    pl_qiskit_circuit = qml.from_qiskit(qcx)
    return pl_qiskit_circuit
