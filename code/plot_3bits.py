import torch
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    updeg = 12
    theoretical_var = torch.load('theoretical_var_3bits.pt', weights_only=True)
    numerical_var = torch.load('numerical_var_3bits.pt', weights_only=True)
    yerr = torch.load('yerr_3bits.pt', weights_only=True)

    plt.figure(figsize=(4.5, 4.5))
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
    # plt.gca().set_aspect('equal')
    # x_vals = 2*np.arange(updeg)
    # ax.errorbar(theoretical_var[0].numpy(), numerical_var[0].numpy(), yerr=yerr[0].numpy(), fmt='-o', capsize=5,
                # label=r'numerical results for $P = X\otimes Z\otimes Z$')
    # ax.errorbar(theoretical_var[1].numpy(), numerical_var[1].numpy(), yerr=yerr[1].numpy(), fmt='-o', capsize=5,
                # label=r'numerical results for $P = X\otimes X\otimes X$')
    # .plot(x_vals, numerical_var[0].numpy(), label=r'numerical results for $P = X\otimes Z\otimes Z$')
    # ax.plot(x_vals, numerical_var[1].numpy(), label=r'numerical results for $P = X\otimes X\otimes X$')
    # plt.scatter(np.power(1e6*np.ones(theoretical_var[0].numpy().shape), theoretical_var[0].numpy()), np.power(1e6*np.ones(theoretical_var[0].numpy().shape), numerical_var[0].numpy()), label=r'numerical results for $P = X\otimes Z\otimes Z$')
    # For case 1, color='#b184dd'. For case 2, color='#F5BD19'
    plt.errorbar(theoretical_var[0].numpy(), numerical_var[0].numpy(), yerr=yerr[0].numpy(), linestyle='none', fmt='o', capsize=4, ecolor='#b184dd', label=r'Monte Carlo', markerfacecolor='#b184dd', markeredgecolor='black', markersize=6)
    plt.plot(theoretical_var[0].numpy(), theoretical_var[0].numpy(), linestyle=':', color='b')
    # ax.scatter(np.log10(theoretical_var[1].numpy()), np.log10(numerical_var[1].numpy()), label=r'theoretical results for $P = X\otimes Z\otimes Z$')
    # ax.plot(x_vals, theoretical_var[1], linestyle=':', label=r'theoretical results for $P = X\otimes X\otimes X$')
    plt.xlabel(r'$\mathrm{Var}_{\theta}^{(\mu_L,\nu_m)}(\rho,O)$ (Polynomial methods)', fontsize=14)
    plt.ylabel(r'$\overline{\mathrm{Var}}_{\theta}^{(\mu_L,\nu_m)}(\rho,O)$ (Monte Carlo)', fontsize=14)
    # plt.xlim([0.028, 0.034])
    # plt.ylim([0.0288, 0.033])
    plt.legend(fontsize=14, markerscale=2, labelspacing=1.5)
    plt.savefig("unitary_1_design_example_3bits_1.pdf", bbox_inches='tight')
    plt.show()
