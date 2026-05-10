import torch
import time
import theoretical_cal

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.complex64

# Support batched input
def trace_normalization_factor(degree, dim):
    if degree == 0:
        return 1
    else:
        dim = torch.tensor(dim)
        return torch.exp(torch.lgamma(dim)) / torch.exp(torch.lgamma(2*degree+dim)) * theoretical_cal.g_fun(degree, dim)

def cal_2th_moment(unit_right, rho_batch, x_batch, updeg, pauli_batch):
    # unit_right size: batch_size, d, d
    # rho: curve numbers, d, d
    # X_L: curve numbers, d, d

    # Expand to broadcast
    unit_exp = unit_right          # (N, 1, d, d)
    dim = unit_exp.shape[-1]

    # Calculation of $\operatorname{Tr}[U\rho U^\dagger X_L]^2$
    tr_temp = torch.matmul(unit_exp, torch.matmul(rho_batch[None, :, :], unit_exp.conj().transpose(-1, -2)))
    tr = torch.matmul(tr_temp, x_batch[None, :, :])
    tr_square = torch.real(torch.diagonal(tr, dim1=-2, dim2=-1).sum(dim=-1)) ** 2

    # Calculation of $\operatorname{Tr}[U\rho U^\dagger P]^{2m}$
    tr2 = torch.matmul(tr_temp, pauli_batch[None, :, :])
    tr2_square = torch.real(torch.diagonal(tr2, dim1=-2, dim2=-1).sum(dim=-1)) ** 2
    exponents = torch.arange(updeg, device=device)
    measure_factor = torch.zeros((updeg), device=device)
    for i in range(len(exponents)):
        measure_factor[i] = trace_normalization_factor(exponents[i], dim)
    tr2_powers = tr2_square.unsqueeze(-1) ** exponents
    return tr_square[:, None] * tr2_powers / measure_factor[None, :]

def gen_angle_range(dim):
    angle_list = torch.zeros(dim**2-1, device=device)
    for i in range(dim**2 - dim):
        angle_list[i] = 2 * torch.pi if i % 2 == 0 else torch.pi / 2
    rows = torch.zeros(dim-1, dtype=torch.int, device=device)
    for j in range(1, dim-1):
        rows[j] = rows[j - 1] + 2 * (dim - j)
    angle_list[rows] = torch.pi
    for i in range(dim**2 - dim, dim**2 - 1):
        i_tran = i - dim**2 + dim + 2
        angle_list[i] = torch.pi * torch.sqrt(torch.tensor(2 * i_tran / (i_tran - 1)))
    return angle_list

def factor(para, dim):
    factor_arr = torch.zeros((dim - 1, para.shape[-1]), device=device)
    for i in range(dim - 1, 0, -1):
        start_pt = 1
        if i != dim - 1:
            start_pt = 1 + 2 * sum(j + 1 for j in range(i, dim - 1))
        const_factor = (torch.pi ** i) * torch.prod(torch.arange(1, i + 1, dtype=torch.float64, device=device)) / 2
        factor_arr[i - 1] = torch.sin(2 * para[start_pt]) * const_factor
        for j in range(1, i):
            if j != i - 1:
                factor_arr[i - 1] *= (torch.cos(para[start_pt + 2 * j]) ** (1 + 2 * j)) * torch.sin(
                    para[start_pt + 2 * j])
            else:
                factor_arr[i - 1] *= (torch.sin(para[start_pt + 2 * j]) ** (1 + 2 * j)) * torch.cos(
                    para[start_pt + 2 * j])
    return torch.prod(factor_arr, dim=0)
def get_unitaries(angles, dim):
    n = angles.shape[-1]
    # initialization
    unit_right_re = torch.eye(dim, device=device).unsqueeze(0)
    unit_right_im = torch.zeros((n, dim, dim), device=device)
    
    for k in range(dim - 1, 0, -1):
        for i in range(k):
            start_pt = (dim + k) * (dim - k - 1)
            theta_temp_even = angles[2 * i + start_pt]
            diag_terms_re = torch.ones((n, dim), device=device)
            diag_terms_im = torch.zeros((n, dim), device=device)
            diag_terms_re[:, 0] = torch.cos(theta_temp_even)
            diag_terms_re[:, 1] = diag_terms_re[:, 0]
            diag_terms_im[:, 0] = torch.sin(theta_temp_even)
            diag_terms_im[:, 1] = -1 * diag_terms_im[:, 0]
            unit_right_re_tmp = unit_right_re * diag_terms_re[:,None,:] - unit_right_im * diag_terms_im[:,None,:]
            unit_right_im_tmp = unit_right_re * diag_terms_im[:,None,:] + unit_right_im * diag_terms_re[:,None,:]
            unit_right_re, unit_right_im = unit_right_re_tmp, unit_right_im_tmp

            theta_temp_odd = angles[2 * i + start_pt + 1]
            odd_mat = torch.zeros((n, dim, dim), device=device)
            odd_mat[:, 0, 0] = torch.cos(theta_temp_odd)
            odd_mat[:, 0, i + 1] = torch.sin(theta_temp_odd)
            odd_mat[:, i + 1, 0] = -odd_mat[:, 0, i+1]
            odd_mat[:, i + 1, i + 1] = odd_mat[:, 0, 0]
            mask = torch.tensor([j for j in range(1, dim) if j != i + 1], device=device)
            odd_mat[:, mask, mask] = 1
            unit_right_re = torch.matmul(unit_right_re, odd_mat)
            unit_right_im = torch.matmul(unit_right_im, odd_mat)

    for s in range(dim**2 - dim, dim**2 - 1):
        cri_index = s - (dim**2 - dim) + 1
        diag_terms_re = torch.ones((n, dim), device=device)
        diag_terms_im = torch.zeros((n, dim), device=device)
        nor_fac = torch.sqrt(torch.tensor(2 / ((cri_index + 1) * (cri_index + 2))))

        mask_less = torch.arange(dim, device=device) < cri_index
        vec_less_re = torch.cos(angles[s] / nor_fac)
        vec_less_im = torch.sin(angles[s] / nor_fac)
        diag_terms_re[:, mask_less] = vec_less_re.unsqueeze(1).repeat(1, cri_index)
        diag_terms_im[:, mask_less] = vec_less_im.unsqueeze(1).repeat(1, cri_index)
        mask_equal = torch.arange(dim, device=device) == cri_index
        vec_equal_re = torch.cos(angles[s] * (cri_index + 1) / nor_fac)
        vec_equal_im = -1 * torch.sin(angles[s] * (cri_index + 1) / nor_fac)
        diag_terms_re[:, mask_equal] = vec_equal_re.unsqueeze(1)
        diag_terms_im[:, mask_equal] = vec_equal_im.unsqueeze(1)

        unit_right_re_tmp = unit_right_re * diag_terms_re[:,None,:] - unit_right_im * diag_terms_im[:,None,:]
        unit_right_im_tmp = unit_right_re * diag_terms_im[:,None,:] + unit_right_im * diag_terms_re[:,None,:]
        unit_right_re, unit_right_im = unit_right_re_tmp, unit_right_im_tmp
    return torch.complex(unit_right_re, unit_right_im)


if __name__ == "__main__":
    num = 10**11
    dim = 8
    batch_size = 4*10**6
    num_batch = num // batch_size
    updeg = 12
    plot_num = 1

    # Initial states
    rho = torch.zeros((dim, dim), dtype=dtype, device=device)
    rho[0, 0] = 1

    # Construction of left unitaries
    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    pauli_y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)

    pauli_word_example = torch.kron(pauli_x, torch.kron(pauli_z, pauli_z))

    obs_example = torch.kron(pauli_z, torch.kron(torch.eye(2, dtype=dtype, device=device), pauli_z))

    lam_fin = torch.zeros((dim, dim), dtype=dtype, device=device)
    for i in range(0, dim - 1):
        lam_fin[i, i] = torch.sqrt(torch.tensor(2.0 / (dim * (dim - 1))))
    lam_fin[-1, -1] = (1 - dim) * torch.sqrt(torch.tensor(2.0 / (dim * (dim - 1))))

    X_L_example = torch.matmul(torch.matmul(pauli_word_example, lam_fin) - torch.matmul(lam_fin, pauli_word_example), obs_example) / torch.sqrt(torch.tensor(2.0))

    X_L_example_dagger = X_L_example.conj().transpose(-1,-2)
    X_L_example_squ = torch.matmul(X_L_example, X_L_example_dagger).real

    # Pauli words in right unitaries
    measure_word = torch.kron(pauli_z, (torch.kron(pauli_z, pauli_z)))
    # measure_word_arr[1] = torch.kron(pauli_x, (torch.kron(pauli_z, pauli_z)))


    start_time = time.perf_counter()
    var_gp_arr = torch.zeros((num_batch, updeg), device=device)
    m2_gp_arr = torch.zeros_like(var_gp_arr)

    angle_list = gen_angle_range(dim)

    theoretical_var = torch.zeros((updeg))
    for i in range(updeg):
        theoretical_var[i] = torch.real(theoretical_cal.theoretical_calculation(X_L_example, measure_word, i))

    for s in range(num_batch):
        if s % 50 == 0:
            print(s, sep=' ', end=' ', flush=True)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")
        alpha_arr = torch.stack([torch.rand(batch_size, device=device) * angle for angle in angle_list])
        env_right = get_unitaries(alpha_arr, dim)
        factor_s = factor(alpha_arr, dim)
        result = cal_2th_moment(env_right, rho, X_L_example, updeg, measure_word) * factor_s[:,None]
        # Calculation of the average over each batch
        var_gp_arr[s] = torch.mean(result, dim=0)
        # Calculation of the variance of Monte Carlo method over each batch
        m2_gp_arr[s] = torch.sum((result - result.mean(dim=0, keepdim=True))**2, dim=0)

    numerical_var = var_gp_arr.mean(dim=0).cpu()

    m2_tmp_arr = m2_gp_arr[0]
    var_tmp_arr = var_gp_arr[0]

    for i in range(1, num_batch):
        delta = var_gp_arr[i] - var_tmp_arr
        m2_tmp_arr = m2_tmp_arr + m2_gp_arr[i] + delta ** 2 * i/(i+1) * batch_size
        var_tmp_arr = var_tmp_arr + delta / (i+1)
    yerr_arr = torch.sqrt(m2_tmp_arr) / num
    yerr_arr = yerr_arr.cpu()
    # yerr = torch.sqrt(yerr_arr.mean(dim=0) / (num * num_batch)).cpu()

    torch.save(theoretical_var, 'theoretical_var_3bits_v3.pt')
    torch.save(numerical_var, 'numerical_var_3bits_v3.pt')
    torch.save(yerr_arr, 'yerr_3bits_v3.pt')