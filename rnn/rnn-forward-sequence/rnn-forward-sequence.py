import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, b_h: np.ndarray) -> dict:
    hidden=h_0.copy()
    states=[]
    for step in range(X.shape[1]):
        hidden=np.tanh(X[:,step,:]@W_xh.T+hidden@W_hh.T+b_h)
        states.append(hidden)
    return {"hidden_states":np.stack(states,axis=1).astype(np.float64),"final_hidden_state":hidden.astype(np.float64)}