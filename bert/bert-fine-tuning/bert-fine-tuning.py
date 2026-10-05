import numpy as np

def bert_fine_tuning_step(hidden_states: np.ndarray, labels: np.ndarray,
                          classifier_W: np.ndarray, classifier_b: np.ndarray,
                          learning_rate: float) -> dict:
    classification_states = hidden_states[:, 0, :]
    logits = classification_states @ classifier_W + classifier_b
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    batch_size = labels.shape[0]
    loss = -np.mean(np.log(probabilities[np.arange(batch_size), labels]))
    grad_logits = probabilities.copy()
    grad_logits[np.arange(batch_size), labels] -= 1.0
    grad_logits /= batch_size
    grad_W = classification_states.T @ grad_logits
    grad_b = np.sum(grad_logits, axis=0)
    return {
        "new_classifier_W": classifier_W - learning_rate * grad_W,
        "new_classifier_b": classifier_b - learning_rate * grad_b,
        "loss": float(loss),
    }