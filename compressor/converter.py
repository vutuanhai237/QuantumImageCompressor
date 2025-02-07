from .compilation import state_to_qc, circuit_curry, cost_curry
import pennylane as qml
import numpy as np

def thetas_x_to_thetas_y(y, thetas_x, num_layers):
	# Convert prepare_y to quantum circuit
	qcy = state_to_qc(y)
	num_qubits = qcy.num_qubits
	circuit_at_y = circuit_curry(qcy, num_qubits, num_layers)
	cost_func_at_y = cost_curry(circuit_at_y)
	grad_func_at_y = qml.grad(cost_func_at_y)

	grad_theta_at_thetas_x_of_y = grad_func_at_y(thetas_x)
	norm2_grad_theta_at_thetas_x_of_y = np.linalg.norm(grad_theta_at_thetas_x_of_y)**2
	cost_at_thetas_x_of_y = cost_func_at_y(thetas_x)

	thetas_y = thetas_x - grad_theta_at_thetas_x_of_y * cost_at_thetas_x_of_y / norm2_grad_theta_at_thetas_x_of_y
	return thetas_y, cost_func_at_y(thetas_y)