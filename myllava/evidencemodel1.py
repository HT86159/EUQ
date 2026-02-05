import torch
import numpy as np
import itertools
import mpmath

class EvidenceModel:
    def __init__(self, Beta):
        self.B = Beta
        # self.bias = bias
        self.evidence_weights = None
        self.w_pos = None
        self.w_neg = None
        self.eta_pos_temp = None
        self.eta_neg_temp = None
        self.K = None

    def get_evidence_weights(self, Phi):
        # import pdb;pdb.set_trace()
        J, K = self.B.shape
        with torch.no_grad():
            B = self.B.to('cuda:0')
            Phi = Phi.to('cuda:0')
            # M = Phi.unsqueeze(0)
            # M = M.squeeze(1)
            M = Phi
            B_star = B - B.mean(axis=1, keepdims=True)
            # print(f"M shape: {M.shape}")
            # print(f"B_star shape: {B_star.shape}")
            # W = ((torch.mm(M, B_star)) / B.shape[0]).expand(J, K)
            W = ((torch.mm(B_star, M)) / B.shape[0]).expand(J, K)
        self.evidence_weights = W.unsqueeze(0).to(dtype=torch.float64)
        self._calculate_basic_terms()
        return self.evidence_weights

    # def get_evidence_weights(self, Phi):
    #     J, K = self.B.shape
    #     with torch.no_grad():
    #         B = self.B.to(torch.float32)
    #         bias=self.bias.to(torch.float32)
    #         Phi = Phi.to(torch.float32)
    #         # M = Phi.unsqueeze(0)
    #         #M = Phi.squeeze(1)
    #         M = Phi
    #         B_star = B - B.mean(axis=1, keepdims=True)
    #         #import pdb;pdb.set_trace()
    #         bias_star=(bias-bias.mean(axis=0,keepdims=True))/B.shape[0]/4
    #         bias_star = bias_star.T
    #         #W = ((torch.mm(M, B_star)) / B.shape[0]).expand(J, K)
    #         W = ((torch.mm(M, B_star)) / B.shape[0])-bias_star
    #         self.evidence_weights = W.unsqueeze(0).to(dtype=torch.float64)
    #         self._calculate_basic_terms()
    #     return self.evidence_weights


    def _calculate_basic_terms(self):
        omega_jk_positive = torch.relu(self.evidence_weights)
        omega_jk_negative = torch.relu(-self.evidence_weights)
        self.w_pos1 = omega_jk_positive.sum(1)[0].unsqueeze(0)
        self.w_neg1 = omega_jk_negative.sum(1)[0].unsqueeze(0)
        self.K = self.w_pos1.shape[-1]
        self.eta_pos_temp = 1 / (torch.exp(self.w_pos1).sum(dim=1) - self.K + 1)
        self.eta_neg_temp = 1 / (1 - torch.prod(1 - torch.exp(-self.w_neg1), dim=1))

        # Handle zeros in w_pos
        sorted_w_pos = torch.sort(self.w_pos1.flatten())[0]
        second_smallest = sorted_w_pos[torch.nonzero(sorted_w_pos > 0, as_tuple=True)[0][0]]
        w_pos1_copy = self.w_pos1.clone()  # 创建 w_pos1 的副本
        w_pos1_copy[w_pos1_copy == 0] = second_smallest  # 修改副本中的值
        self.w_pos2 = w_pos1_copy  # 将修改后的副本赋值给 w_pos2
        # Handle zeros in w_neg
        sorted_w_neg = torch.sort(self.w_neg1.flatten())[0]
        second_smallest = sorted_w_neg[torch.nonzero(sorted_w_neg > 0, as_tuple=True)[0][0]]
        w_neg1_copy = self.w_neg1.clone()  # 创建 w_neg1 的副本
        w_neg1_copy[w_neg1_copy == 0] = second_smallest  # 修改副本中的值
        self.w_neg2 = w_neg1_copy  # 将修改后的副本赋值给 w_neg2
        # Calculate kappa
        self.kappa = torch.sum(self.eta_pos_temp.reshape(-1, 1) * (torch.exp(self.w_pos2) - 1) * (1 - self.eta_neg_temp.reshape(-1, 1) * torch.exp(-self.w_neg1)), dim=1)
        self.eta_temp = 1 / (1 - self.kappa)

    def get_evidence_conflict(self):
        return self.kappa

    # def get_evidence_ignorance(self):
    #     # Calculate ignorance value using precomputed terms
    #     exp_results = torch.exp(-torch.sum(self.w_neg2, dim=1))
    #     w_neg_sum = exp_results.sum()
    #     ig = self.eta_temp * self.eta_pos_temp * self.eta_neg_temp * w_neg_sum
    #     return ig

    def get_evidence_ignorance(self):
        # 设置 mpmath 的精度
        mpmath.mp.dps = 50  # 50 位十进制精度（大约相当于 128 位二进制精度）
        
        w_neg2_flatten = self.w_neg2.flatten()  # 展平为一维张量
        w_neg2_mp = [mpmath.mpf(x.item()) for x in w_neg2_flatten]

        eta_temp_mp = mpmath.mpf(self.eta_temp.item())
        eta_pos_temp_mp = mpmath.mpf(self.eta_pos_temp.item())
        eta_neg_temp_mp = mpmath.mpf(self.eta_neg_temp.item())

        # 使用 mpmath 进行高精度计算
        exp_results_mp = [mpmath.exp(-x) for x in w_neg2_mp]
        w_neg_sum_mp = sum(exp_results_mp)

        # 计算 ignorance value
        ig_mp = (
            eta_temp_mp *
            eta_pos_temp_mp *
            eta_neg_temp_mp *
            w_neg_sum_mp
        )

        # 将结果转换回浮点数
        ig = float(ig_mp)

        return ig

    def get_nonspecific(self):
        # Calculate non-specificity value using precomputed terms
        eta_mul = self.eta_temp * self.eta_pos_temp * self.eta_neg_temp
        prod_term = torch.prod(1 - torch.exp(-self.w_neg2), dim=1, keepdim=True) / (1 - torch.exp(-self.w_neg2))
        second_term = (torch.exp(self.w_pos1) - 1) + prod_term
        first_term = eta_mul.reshape(-1, 1) * torch.exp(-self.w_neg2)
        m_theta = first_term * second_term
        return 1 - m_theta.sum()

    def compute_m(self, labels):
        w_neg = self.w_neg1.squeeze()
        w_pos = self.w_pos1.squeeze()
        mass_function = dict()
        subsets = self.generate_subsets(labels)
        for theta_set in subsets:
            if len(theta_set) == 0:
                m_value = 0
            elif len(theta_set) == 1:
                k = theta_set[0]
                m_value = torch.exp(-w_neg[k]) * (torch.exp(w_pos[k]) - 1 + torch.prod(torch.tensor([1 - torch.exp(-w_neg[l]) for l in range(len(w_neg)) if l != k])))
            elif len(theta_set) > 1:
                prod_not_in_A = np.prod([1 - np.exp(-w_neg[k]) for k in range(len(w_neg)) if k not in theta_set])
                prod_in_A = np.prod([np.exp(-w_neg[k]) for k in theta_set])
                m_value = prod_not_in_A * prod_in_A
            mass_function[theta_set] = m_value
        values = list(mass_function.values())
        total = sum(values)
        mass_function = {key: value / total for key, value in mass_function.items()}
        return mass_function

    def generate_subsets(self, input_set):
        subsets = []
        for r in range(len(input_set) + 1):
            subsets.extend(itertools.combinations(input_set, r))
        return [tuple(subset) for subset in subsets]

    def pl_A(self, labels, m_values):
        subsets = self.generate_subsets(labels)
        pl = dict()
        for A in subsets:
            pl_value = 0
            for B, m_B in m_values.items():
                if set(B).intersection(A):
                    pl_value += m_B
            pl[A] = pl_value
        return pl