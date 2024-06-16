# Utilities for converting floating point numbers to fixed point numbers
import numpy as np

def _scaling_funct(params):
    '''Find optimal scaling function for float- to fixed-point mapping.
    
    Parameter
    ---------
    params : dict
        Dictionary containing information required for float- to fixed-point mapping
        
    Returns
    ------
    scaling_funct : callable
        Optimal scaling function for float- to fixed-point conversion
    ''' 
    # Sort parameters based on the maximum absolute value.
    sorted_params = dict(sorted(params.items(), key=lambda x: np.max(np.abs(x[1]['val'])), reverse=True))
    
    # Initialize scaling function.
    scaling_funct = None
    
    for key, val in sorted_params.items():
        # Check if the current parameter is signed or unsigned.
        if val['signed'] == 's':
            signed_shift = 1
        else:
            signed_shift = 0
        
        # Check if the maximum absolute value of the current parameter is equal to the maximum value.
        # 
        if np.max(val['val']) == np.max(np.abs(val['val'])):
            max_abs = True
            max_abs_val = np.max(val['val'])
        else:
            max_abs = False
            max_abs_val = np.max(np.abs(val['val']))
        
        # Find the maximum representable value for the current parameter.
        if max_abs:
            rep_val = 2**(val['bits'] - signed_shift) - 1
        else:
            rep_val = 2**(val['bits'] - signed_shift)
        
        # Find the maximum shift value for the current parameter.
        max_shift = np.max(val['shift'])
        
        # Calculate the maximum representable value for the current parameter after shifting.
        max_rep_val = rep_val * 2**max_shift
      
        if scaling_funct:
            # Scale the current parameter using the scaling function.
            scaled_vals = scaling_funct(val['val'])

            # Find the maximum absolute value of the scaled parameter.
            max_abs_scaled_vals = np.max(np.abs(scaled_vals))

            if max_abs_scaled_vals <= max_rep_val:
                # If the current parameter is already within the representable range, skip.
                continue
            else:
                # If the current parameter is not within the representable range, update the scaling function.
                p1 = max_rep_val
                p2 = max_abs_val
        
        else:
            # If the scaling function is not initialized, initialize it with the current parameter.
            p1 = max_rep_val
            p2 = max_abs_val         
        
        # Update scaling function based on the current parameter.
        scaling_funct = lambda x: p1 / p2 * x
        
    return scaling_funct

class LIFFixedParams:
    def __init__(self, v_th, bias_mant, bias_exp, weights, weight_exp):
        self.v_th = v_th
        self.bias_mant = bias_mant
        self.bias_exp = bias_exp
        self.weights = weights
        self.weight_exp = weight_exp

    def __repr__(self):
        # Return dictionary representation of the object.
        return f"LIFFixedParams(\n\tv_th={self.v_th}, \n\tbias_mant={self.bias_mant}, bias_exp={self.bias_exp}, \n\tweights=\n{self.weights}, \n\tweight_exp={self.weight_exp})"
    

def float2fixed_lif_parameter(lif_params) -> LIFFixedParams:
    '''Float-to-fixed-point mapping for LIF parameters.
    
    Parameters
    ---------
    lif_params : dict
        Dictionary with parameters for LIF network with floating-point ProcModel
        
    Returns
    ------
    lif_params_fixed : LIFFixedParams
        Dictionary with parameters for LIF network with fixed-point ProcModel
    '''
    
    # Get the optimal scaling function for float-to-fixed-point mapping.
    scaling_funct = _scaling_funct(lif_params)
    
    # Find the optimal fixed-point representation for each parameter.

    # Bias parameter
    # bias_mant_bits = lif_params['bias']['bits']
    # scaled_bias = scaling_funct(lif_params['bias']['val'])[0]
    # bias_exp = int(np.ceil(np.log2(scaled_bias) - bias_mant_bits + 1))
    # if bias_exp <=0:
    #     bias_exp = 0
    
    # Weights parameter
    weight_mant_bits = lif_params['weights']['bits']    
    scaled_weights = np.round(scaling_funct(lif_params['weights']['val']))
    # Check if we need this?
    weights_exp = 0
    # weight_exp = int(np.ceil(np.log2(scaled_bias) - weight_mant_bits + 1))
    # weight_exp = np.max(weight_exp) - 6
    # if weight_exp <=0:
    #     diff = weight_exp
    #     weight_exp = 0

    # Convert the scaled parameters to fixed-point representation.
    # bias_mant = int(scaled_bias // 2**bias_exp)
    weights = scaled_weights.astype(np.int32)
    
    # Build an object with fixed-point parameters.
    lif_params_fixed = LIFFixedParams(
        v_th=int(scaling_funct(lif_params['vth']['val']) // 2**lif_params['vth']['shift'][0]),
        bias_mant=0,
        bias_exp=0,
        weights=np.round(scaled_weights / (2 ** lif_params['weights']['shift'][0])).astype(np.int32),
        weight_exp=weights_exp
    )
    
    return lif_params_fixed

def scaling_funct_dudv(val):
    '''Scaling function for du, dv in LIF
    ''' 
    assert val < 1, 'Passed value must be smaller than 1'
    
    return np.round(val * 2 ** 12).astype(np.int32)


""" 
After having defined some primitive conversion functionality we next convert the parameters for the critical network.

    To constrain the values that we need to represent in the bit-accurate model, we have to find the dynamical range 
of the state parameters of the network, namely u and v of the LIF neurons. 

    We note that for both variables the distributions attain large (small) values with low probability. We hence will remove them in the dynamical range 
to increase the precision of the overall representation. We do so by choosing: 0.2 and 0.8 quantiles as minimal resp. maximal values for the dynamic ranges.

    We finally also need to pass some information about the concrete implementation, e.g. the precision and the bit shifts performed.
"""

""" u_low = np.quantile(data_u_critical.flatten(), 0.2)
u_high = np.quantile(data_u_critical.flatten(), 0.8)
v_low = np.quantile(data_v_critical.flatten(), 0.2)
v_high = np.quantile(data_v_critical.flatten(), 0.8) """

class LIFFloat2FixedParams:
    def __init__(self, u_interval: np.ndarray, v_interval: np.ndarray, weights: np.ndarray, v_th: float = 1.0, bias: float = 0):
        '''
        @param u_interval: np.ndarray
            The interval of the u values [u_min, u_max]
        @param v_interval: np.ndarray
            The interval of the v values [v_min, v_max]
        @param weights: np.ndarray
            The weights of the network
        @param v_th: float
            The threshold of the network (default 1.0)
        @param bias: float
            The bias of the network
        '''
        self.u_interval = u_interval
        self.v_interval = v_interval
        self.weights = weights
        self.v_th = v_th
        self.bias = bias

def lif_params_float2fixed(lifParams: LIFFloat2FixedParams) -> LIFFixedParams:
    weights = lifParams.weights
    bias = lifParams.bias

    params = {'vth': {'bits': 17, 'signed': 'u', 'shift': np.array([6]), 'val': np.array([lifParams.v_th])},
            'u': {'bits': 24, 'signed': 's', 'shift': np.array([0]), 'val': np.array(lifParams.u_interval)},
            'v': {'bits': 24, 'signed': 's', 'shift': np.array([0]), 'val': np.array(lifParams.v_interval)},
            'bias': {'bits': 13, 'signed': 's', 'shift': np.arange(0, 3, 1), 'val': np.array([bias])},
            'weights' : {'bits': 8, 'signed': 's', 'shift': np.arange(6,22,1), 'val': weights}}

    mapped_params = float2fixed_lif_parameter(params)
    return mapped_params

# In the Bit accurate model, du and dv are 12-bit unsigned integers (precision of 12 bits on a 16-int variable
PRECISION_DU_DV = 12

def scaling_dudv(val):
    '''Scaling function for du, dv in LIF
    ''' 
    assert val < 1, 'Passed value must be smaller than 1'
    
    # return np.round(val * (2 ** PRECISION_DU_DV)).astype(np.int32)
    return np.round( val * ((2 ** PRECISION_DU_DV) - 1) ).astype(np.int32)