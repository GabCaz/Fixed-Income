''' File with classes to perform tree operations on securities
 (ie fit a tree to bond prices, price securities using the tree, etc) '''

import numpy as np
import pandas as pd

class RateTree:
    ''' Class for interest rate tree pricing, allowing to:
        *** fit a tree. Possibilities: Ho and Lee (fit to bond yields), full Black, Derman
            and Toy, a.k.a. BDT (fit to yields and volatilities), Hull and White (include a mean-reversion
            parameter)
        *** price various securities depending on interest rates '''

    def __init__(self, prices, mat, sig, q=.5, face_value=1, r_tree=None, kappa=None):
        ''' Create a tree object
            Input:
            *** prices: a numpy array of zero-coupon bond prices, all normalized to have final payoff 1
                        (ie target discount factors)
            *** mat: a numpy array of corresponding maturities (in years)
            *** sig: array-like (for BDT model) or float (for Ho and Lee) of corresponding bond return volatilities
                    In the case of BDT, it is the standard deviation of log rates
            *** q: RNP of "up interest rate"
            *** kappa: the mean-reversion parameter if the model allows for one (eg Hull and White) '''
        self.prices = prices
        self.mat = mat
        self.sig = sig
        self.q = q
        self.fv = face_value
        self.kappa = kappa
        if r_tree is None:
            self.r_tree = np.zeros((mat.size, mat.size))  # tree with the interest rates
        else:
            self.r_tree = r_tree

    def security(self, sec):
        ''' /!\ Call after fitting the interest rate tree
            Purpose: given a security, prices the security, and gives its duration, with the current tree
            Inputs:
            *** sec: object with a method allowing you to get the cash-flow generated
                by that security at any point in the tree (_get_raw_cf_mat)
            Returns:
            *** tuple: (price, duration) '''
        # get the payoff matrix of the security, for the current self.r_tree
        raw_payoff_mat = sec.get_raw_cf_mat(self.r_tree)
        # discount these payoffs back to now, exercising early if optimal
        present_value_tree = self._discount_payoffs(raw_payoff_mat, early=sec.early, sec=sec)
        price = present_value_tree[0, 0]
        ir_delta = (present_value_tree[0, 1] - present_value_tree[1, 1]) / (self.r_tree[0, 1] - self.r_tree[1, 1])
        spot_rate_dur = -ir_delta / price
        summary = {'Price': price,
                   'Spot duration': spot_rate_dur}
        print(pd.DataFrame(summary, index=['Security']))
        return (price, spot_rate_dur)

    def _fit_ho_lee(self, m_min=0, m_max=1, accuracy=1e-10):
        ''' fit a Ho and Lee tree to the given data. Possibly pass in min and max possible values for m
            /!\ Assumes the "vol" attribute is a flot (not an array-like) '''
        for k in range(self.mat.size):
            self._fit_for_p_hl(k, m_min=0, m_max=1, accuracy=1e-10)

    def _discount_payoffs(self, p_mat, early=False, sec=None):
        ''' Inputs:
            *** p_mat: a matrix of payoffs through time (each column corresponding to the payoff
                at a maturity (the corresponding maturity in self.mat) and each element in this column
                corresponding to the payoff at that point in the tree)
            *** early: True iif we should take into account possible early exercise when discounting (ie
                the security has an American feature)
            *** sec: a security object. This will only be useful if early=True (to get the exercise
                value at each point), otherwise the p_mat is theoretically enough to get the price
            Outputs:
            *** the discounted cash-flows (ie the matrix of prices at different moments) '''
        for time in reversed(range(p_mat.shape[0] - 1)):  # for each previous time, going backwards...
            #             interv = # time interval in this layer of the tree. TODO TO HANDLE LOWER TIMESTEP
            # the value at the previous layer, in the corresponding node
            for node in range(time + 1):
                p_mat[node, time] += (self.q * p_mat[node, time + 1] +
                                      (1 - self.q) * p_mat[node + 1, time + 1]) / (1 + self.r_tree[node, time])
                # If you can act early and this action decreases the present value of amounts due,
                # then exercise early
                if early:
                    val_if_exercise_early = sec.cf_early_exercise(time, self.r_tree[node, time])
                    #                     print('EARLY EXERCISE')
                    #                     print('time is', time)
                    #                     print('Value exercised early is', val_if_exercise_early)
                    #                     print('Value without early exercise was', p_mat[node, time])
                    if sec.exercise_early(p_mat[node, time], val_if_exercise_early, time, self.r_tree[node, time]):
                        p_mat[node, time] = val_if_exercise_early
        return p_mat

    def print_tree(self):
        ''' prints the fitted tree '''
        to_display = pd.DataFrame(self.r_tree)
        to_display.columns = self.mat
        idx = ["-"] * self.r_tree.shape[0]
        idx[0] = "More up branches"
        idx[-1] = "More down branches"
        to_display.index = idx
        print(to_display)

    def _bond_price_with_tree(self, k, get_up_down=False, zcb_nom=1):
        '''
        Inputs:
        *** k: position in the arrays mat and prices, corresponding to the bond we want to price
        *** get_up_down: get the price of the bond in the down and up state, allowing to fit BDT
        Outputs:
        *** price of the k-th bond, resulting from discounting with the given tree
        *** optionally, also returns up and down price
        '''
        zcb_mat = np.zeros((self.mat[k] + 1, self.mat[k] + 1))  # np array containing the price of the bond at each node
        zcb_mat[:, -1] = zcb_nom * np.ones(self.mat[k] + 1)  # prices at maturity, ie time = maturity
        disc_mat = self._discount_payoffs(zcb_mat)
        if get_up_down:
            return disc_mat[0, 0], disc_mat[0, 1], disc_mat[1, 1]
        return disc_mat[0, 0]

    def _parametrized_layer_hl(self, m, k):
        ''' HO, LEE
        given an value for m and k, gives next layer on the tree,
        corresponding to that given m.
        Input:
        *** m: the Ho Lee "drift" for interest rates
        *** k: the layer (time) to fill, e.g. k = 1 to fill the two rates after the first two branches.
        Will change the r_tree in place '''
        self.r_tree[0, k] = self.r_tree[0, k - 1] + self.sig + m
        for node in range(1, k + 1):
            self.r_tree[node, k] = self.r_tree[0, k] - 2 * node * self.sig

    def _fit_for_p_hl(self, k, m_min=0, m_max=1, accuracy=1e-10):
        ''' HO, LEE
            Given a position k in the prices and mat arrays, fits the rate tree layer
            corresponding to these, granted the previous layers have already been fitted '''
        target = self.prices[k]
        while abs(m_max - m_min) > accuracy:
            mid = (m_max + m_min) / 2
            self._parametrized_layer_hl(mid, k)
            error_mid = self._bond_price_with_tree(k) - target
            self._parametrized_layer_hl(m_max, k)
            error_high = self._bond_price_with_tree(k) - target
            if error_high * error_mid > 0:
                m_max = mid
            else:
                m_min = mid
        return mid

    def _fit_bdt(self):
        from scipy.optimize import minimize
        ''' fit a Ho and Lee tree to the given data. Possibly pass in min and max possible values for m
            /!\ Assumes the "vol" attribute is a flot (not an array-like) '''
        for k, p in enumerate(self.prices):
            mini = minimize(self._error_fit_bdt, [.1, 0], args=(k,))

    def _parametrized_layer_bdt(self, r_bar, bdt_sig, k):
        ''' BLACK, DERMAN, TOY
        Inputs:
        *** r_bar (float): value of the maximum interest rate (on the top of the tree)
        *** bdt_sig (float): the 'BDT tree sigma' of that layer, ie the volatility of the log rates at that level
        *** k: the layer (time) to fill, e.g. k = 1 to fill the two rates after the first two branches.
        '''
        self.r_tree[0, k] = r_bar  # the maximum rate at that time
        for node in range(1, k + 1):
            self.r_tree[node, k] = r_bar * np.exp(-2 * node * bdt_sig)

    def _error_fit_bdt(self, x0, k):
        ''' BLACK, DERMAN, TOY
            Returns a measure of fit error to the BDT model for the k_th (price, volatility), used to fit
            the BDT tree
            INPUTS:
            *** x0: array-like with the r_bar and bdt_sig to create layer'''
        # Retrieve arguments
        r_bar = x0[0]
        bdt_sig = x0[1]
        # make layer with the given parameters
        target_p = self.prices[k]
        # Price error
        self._parametrized_layer_bdt(r_bar, bdt_sig, k)
        tree_price, tree_down, tree_up = self._bond_price_with_tree(k, get_up_down=True)
        error_price = tree_price - target_p  # difference in price
        # Error vol
        tree_sig = 0
        target_sig = 0
        if k >= 1:
            target_sig = self.sig[k - 1]
            r_d = pow(self.fv / tree_up,
                      1 / (self.mat[k] - 1)) - 1  # return in the up state (ie k-year rate in the up-state)
            r_u = pow(self.fv / tree_down, 1 / (self.mat[k] - 1)) - 1
            tree_sig = np.log(r_u / r_d) / 2
        error_vol = 100 * (target_sig - tree_sig)  # difference in vols
        # TODO: adjust the tree to handle other timesteps
        return pow(error_vol, 2) + pow(error_price, 2)