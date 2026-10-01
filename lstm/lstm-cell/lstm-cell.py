import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> dict:
    joined=np.concatenate([h_prev,x_t],axis=-1)
    f=sigmoid(joined@W_f.T+b_f)
    i=sigmoid(joined@W_i.T+b_i)
    candidate=np.tanh(joined@W_c.T+b_c)
    o=sigmoid(joined@W_o.T+b_o)
    cell=f*C_prev+i*candidate
    hidden=o*np.tanh(cell)
    return {"hidden_state":hidden.astype(np.float64),"cell_state":cell.astype(np.float64)}