'''
Below is a Jupyter Notebook that walks you through implementing the `EvidenceModel` and a neural network in PyTorch. It starts by importing necessary libraries, creating the neural network, and continues with the creation of the evidence model and plotting functions.

First, let's start by importing the libraries we need:
'''

import torch
import numpy as np
import itertools
import mpmath


class EvidenceModel:
    def __init__(self, model_weight, model_bias):
        '''
        input:
            Beta (B): a neural network model head paramenters: (hidden layer; output layer)
            Phi (M):  feature vectors: (hidden layer; 1)
        '''
        self.B = model_weight
        self.bias = model_bias.to("cuda:0")
        self.evidence_weights = None
        self.w_pos = None
        self.w_neg = None
        self.eta_pos_temp = None
        self.eta_neg_temp = None
        self.K = None
        self.weight = None


    def get_evidence_weights(self,Phi):
        # import pdb;pdb.set_trace()
        J, K = self.B.shape # J: hidden layer; K: output layer
        with torch.no_grad():
            B = self.B.to('cuda:0')
            Phi = Phi.to('cuda:0')
            M = Phi.reshape(J,1) #(hidden layer; 1)
            # M = torch.mean(M, dim=0, keepdims=True)
            B_star = B - B.mean(axis=1, keepdims=True) #对[10,3]的10进行归一化,这里是[10,3]-[10]
            # import pdb;pdb.set_trace()
            bias_star = self.bias - self.bias.mean()
            alpha_star = (bias_star + torch.mm(M.T,B_star))/J - M*B_star
            # 本质上只用了(bias_star + torch.mm(M,B_star))/J
            self.evidence_weights = (Phi*B_star) + alpha_star
        self._calculate_basic_terms(Phi)
        return self.evidence_weights

    def _calculate_basic_terms(self,phi):
        omega_jk_positive = torch.relu(self.evidence_weights)
        omega_jk_negative = torch.relu(-self.evidence_weights)
        import pdb;pdb.set_trace()
        self.w_pos1 = omega_jk_positive.sum(0) #torch.Size([output layer])
        self.w_neg1 = omega_jk_negative.sum(0) #torch.Size([output layer])
        # import pdb;pdb.set_trace()
        self.K = self.w_pos1.shape[-1]
        # Handle zeros in w_pos
        second_smallest = 10e-6
        w_pos1_copy = self.w_pos1.clone()  # 创建 w_pos1 的副本
        w_pos1_copy[w_pos1_copy == 0] = second_smallest  # 修改副本中的值
        self.w_pos2 = w_pos1_copy  # 将修改后的副本赋值给 w_pos2
        # Handle zeros in w_neg
        w_neg1_copy = self.w_neg1.clone()  # 创建 w_neg1 的副本
        second_smallest = 10e-6
        w_neg1_copy[w_neg1_copy == 0] = second_smallest  # 修改副本中的值
        self.w_neg2 = w_neg1_copy  # 将修改后的副本赋值给 w_neg2


        self.eta_pos_temp = 1 / (torch.exp(self.w_pos1).sum() - self.K + 1)
        self.eta_neg_temp = 1 / (1 - torch.prod(1 - torch.exp(-self.w_neg1)))

        # Calculate kappa
        # import pdb;pdb.set_trace()
        self.kappa = torch.sum(self.eta_pos_temp.reshape(-1, 1) * (torch.exp(self.w_pos2)-1 ) * (1- self.eta_neg_temp.reshape(-1, 1) * torch.exp(-self.w_neg1)), dim=1)
        self.eta_temp = 1 / (1 - self.kappa)

    def get_evidence_conflict(self):
        if torch.isnan(self.kappa).all():
            return torch.tensor((1))
        return self.kappa

    def get_evidence_ignorance(self):
        # Calculate ignorance value using precomputed terms
        m_pos_omage = 1 / (torch.sum(torch.exp(self.w_pos2)) - self.K + 1)
        m_neg_omage = torch.exp(-torch.sum(self.w_neg1)) / 1 - torch.prod(1 - torch.exp(-self.w_neg1))
        ig = m_pos_omage * m_neg_omage / (1-self.kappa)
        if torch.isnan(ig).all():
            return torch.tensor((1))
        return ig

    def get_nonspecific(self):
        # import pdb;pdb.set_trace()
        eta = 1 / (1 - self.kappa)
        m_theta = eta * self.eta_pos_temp * self.eta_neg_temp 
        m_theta = m_theta * torch.exp(-self.w_neg1) 
        m_theta = m_theta * (torch.exp(self.w_pos2) - 1 + torch.prod(1 - torch.exp(-self.w_neg2))/(1 - torch.exp(-self.w_neg2)))
        nonspecific = 1 - m_theta.sum()
        if torch.isnan(nonspecific).all():
            return torch.tensor((1))
        return nonspecific

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
                # import pdb;pdb.set_trace()
                prod_not_in_A = np.prod([1 - np.exp(-w_neg[k].cpu()) for k in range(len(w_neg)) if k not in theta_set])
                prod_in_A = np.prod([np.exp(-w_neg[k].cpu()) for k in theta_set])
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