import numpy as np

def lstm_forward(X: np.ndarray, W_f: np.ndarray, W_i: np.ndarray,
                 W_c: np.ndarray, W_o: np.ndarray, b_f: np.ndarray,
                 b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray,
                 W_y: np.ndarray, b_y: np.ndarray) -> dict:
    batch,steps,_=X.shape
    hidden=np.zeros((batch,b_f.shape[0]),dtype=np.float64)
    cell=np.zeros_like(hidden)
    outputs=[]
    sigmoid=lambda value:1.0/(1.0+np.exp(-np.clip(value,-500,500)))
    for step in range(steps):
        joined=np.concatenate([hidden,X[:,step,:]],axis=-1)
        forget=sigmoid(joined@W_f.T+b_f)
        input_gate=sigmoid(joined@W_i.T+b_i)
        candidate=np.tanh(joined@W_c.T+b_c)
        output_gate=sigmoid(joined@W_o.T+b_o)
        cell=forget*cell+input_gate*candidate
        hidden=output_gate*np.tanh(cell)
        outputs.append(hidden@W_y.T+b_y)
    return {"outputs":np.stack(outputs,axis=1).astype(np.float64),"final_hidden_state":hidden.astype(np.float64),"final_cell_state":cell.astype(np.float64)}