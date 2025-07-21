import torch
import editdistance

class Evaluator:
    def __init__(self, idx2char={}, blank_index=0, use_levenshtein=True):
        self.idx2char = idx2char
        self.blank_index = blank_index
        self.use_levenshtein = use_levenshtein
        self.reset()

    def reset(self):
        self.total_chars = 0
        self.correct_chars = 0
        self.correct_seqs = 0
        self.total_samples = 0
        self.lev_dists = []

    def greedy_decode(self, logits):
        # logits: [B, T, C]
        predictions = torch.argmax(logits, dim=-1)  # [B, T]
        decoded = []

        for prediction in predictions:
            prev = self.blank_index
            chars = []
            for idx in prediction:
                idx = idx.item()
                if idx != self.blank_index and idx != prev:
                    chars.append(self.idx2char[idx])
                prev = idx
            decoded.append("".join(chars))
        return decoded
    
    def greedy_decode_idx(self, logits):
        predictions = torch.argmax(logits, dim=2)
        predictions= predictions.transpose(0, 1)
        final_predictions = []
        #iterate for each prediction array in the batch
        for prediction in predictions:
            before = 0
            reduced = []
            for t_index in prediction:
                t_index = t_index.item()
                if t_index != 0 and t_index != before:
                    #append the index only if it is not zero and it is different than before
                    reduced.append(t_index)
                before = t_index
            final_predictions.append(reduced)
        return final_predictions

    def update(self, logits, target_strs):
        # logits: [B, T, vocab_size]
        pred_strs = self.greedy_decode(logits)

        for pred, true in zip(pred_strs, target_strs):
            self.total_samples += 1
            self.total_chars += len(true)
            correct = sum(p == t for p, t in zip(pred, true))
            self.correct_chars += correct
            if pred == true:
                self.correct_seqs += 1
            if self.use_levenshtein:
                self.lev_dists.append(editdistance.eval(pred, true))

    def update_baseline(self, logits, labels):
        
        final_predictions = self.greedy_decode_idx(logits)
        for pred_idx_list, label in zip(final_predictions, labels):
            label_list = label.tolist()
            if pred_idx_list == label_list:
                self.correct_seqs +=1

            self.total_samples += 1
            self.total_chars += len(label)
            correct = 0
            for pred_idx, label_idx in zip(pred_idx_list, label):
                if pred_idx == label_idx:
                    correct += 1
            self.correct_chars += correct

    def compute(self):
        char_acc = self.correct_chars / self.total_chars if self.total_chars > 0 else 0.0
        seq_acc = self.correct_seqs / self.total_samples if self.total_samples > 0 else 0.0
        lev_dist = sum(self.lev_dists) / self.total_samples if self.lev_dists else 0.0
        return {
            "char_accuracy": char_acc,
            "seq_accuracy": seq_acc,
            "avg_levenshtein": lev_dist if self.use_levenshtein else None
        }

    def print(self):
        metrics = self.compute()
        print(f"Character accuracy:  {metrics['char_accuracy']:.4f}")
        print(f"Sequence accuracy:   {metrics['seq_accuracy']:.4f}")
        if self.use_levenshtein:
            print(f"Avg Levenshtein:     {metrics['avg_levenshtein']:.2f}")