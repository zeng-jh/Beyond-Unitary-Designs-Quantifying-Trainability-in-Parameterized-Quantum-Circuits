import torch
import time
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.complex64


# Calculation for only one element
def g_fun(x, dim):
    prod_arr = dim + 2*torch.arange(x, device='cpu')
    prod = torch.prod(prod_arr)
    return torch.exp(torch.lgamma(2*x + 1)) / (2**x * torch.exp(torch.lgamma(x+1))) * prod

def theoretical_calculation(x_patch, pauli_patch, maxdeg):
    dim = x_patch.shape[-1]
    coeff_1 = 2 * maxdeg / (dim * (2*maxdeg+dim+1)*(dim+2))
    coeff_2 = (2*maxdeg+dim+2) / (dim * (2*maxdeg+dim+1)*(dim+2))
    px_patch = torch.matmul(pauli_patch, x_patch)
    term_1 = px_patch.diagonal(dim1=-2, dim2=-1).sum(-1) ** 2
    term_2 = torch.matmul(px_patch, px_patch).diagonal(dim1=-2, dim2=-1).sum(-1)
    term_3 = torch.matmul(x_patch, x_patch).diagonal(dim1=-2, dim2=-1).sum(-1)
    return coeff_1 * (term_1+term_2) + coeff_2 * term_3


if __name__ == "__main__":
    dim = 8
    updeg = 12
    plot_num = 2

    # Initial states
    rho_arr = torch.zeros((2, dim, dim), dtype=dtype, device=device)
    # rho_arr[0,0,0] = 1/2
    # rho_arr[0,0,-1] = 1/2
    # rho_arr[0,-1,0] = 1/2
    # rho_arr[0,-1,-1] = 1/2
    rho_arr[0,0,0] = 1
    rho_arr[1,0,0] = 1

    # Construction of left unitaries
    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    pauli_y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    pauli_I = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)

    pauli_word_example = torch.kron(pauli_x, torch.kron(pauli_z, pauli_z))

    obs_example = torch.kron(pauli_z, torch.kron(torch.eye(2, dtype=dtype, device=device), pauli_z))

    lam_fin = torch.zeros((dim, dim), dtype=dtype, device=device)
    for i in range(0, dim - 1):
        lam_fin[i, i] = torch.sqrt(torch.tensor(2.0 / (dim * (dim - 1))))
    lam_fin[-1, -1] = (1 - dim) * torch.sqrt(torch.tensor(2.0 / (dim * (dim - 1))))

    X_L_arr = torch.zeros_like(rho_arr)
    X_L_example = torch.matmul(torch.matmul(pauli_word_example, lam_fin) - torch.matmul(lam_fin, pauli_word_example),
                               obs_example) / torch.sqrt(torch.tensor(2.0))
    for k in range(plot_num):
        X_L_arr[k] = X_L_example
    X_L_arr_dagger = X_L_arr.conj().transpose(-1, -2)
    X_L_arr_squ = torch.matmul(X_L_arr, X_L_arr_dagger).real

    # Pauli words in right unitaries
    measure_word_arr = torch.zeros_like(rho_arr)
    measure_word_arr[0] = torch.kron(pauli_z, (torch.kron(pauli_z, pauli_z)))
    measure_word_arr[1] = torch.kron(pauli_x, (torch.kron(pauli_z, pauli_z)))

    theoretical_results = torch.zeros((X_L_arr.shape[0], updeg))
    for i in range(updeg):
        theoretical_results[:, i] = torch.real(theoretical_calculation(X_L_arr, measure_word_arr, i))

    plt.figure()
    plt.plot(theoretical_results[0], color='blue')
    plt.plot(theoretical_results[1], color='red')
    plt.show()