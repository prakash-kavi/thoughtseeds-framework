import json
import os
import numpy as np
import itertools
from config.meditation_config import THOUGHTSEED_INTERACTIONS, THOUGHTSEED_AGENTS

class LearningDataExporter:
    """Exports training results to JSON files"""
    
    def __init__(self, experience_level, thoughtseeds, states):
        self.experience_level = experience_level
        self.thoughtseeds = thoughtseeds
        self.states = states
        
        # Create output directories
        os.makedirs('./results/data', exist_ok=True)
        os.makedirs('./results/plots', exist_ok=True)
    
    def export_all(self, weights, transition_analyzer, state_history, 
                  activations_history, meta_awareness_history, dominant_ts_history):
        """Export all training data to JSON files"""
        self.export_weights(weights)
        self.export_transition_stats(transition_analyzer)
        self.export_learning_history(state_history, activations_history, 
                                    meta_awareness_history, dominant_ts_history)
        self.export_parameter_files(weights, transition_analyzer, state_history,
                                   activations_history, meta_awareness_history)
        
        print(f"Data export complete for {self.experience_level} experience level.")
    
    def export_weights(self, weights):
        """Export learned weights to JSON"""
        with open(f"./results/data/learned_weights_{self.experience_level}.json", "w") as f:
            json.dump({
                "weights": weights.tolist() if isinstance(weights, np.ndarray) else weights,
                "thoughtseeds": self.thoughtseeds,
                "states": self.states
            }, f, indent=2)
        print(f"  - Weights saved to ./results/data/learned_weights_{self.experience_level}.json")
    
    def export_transition_stats(self, analyzer):
        """Export transition statistics to JSON"""
        with open(f"./results/data/transition_stats_{self.experience_level}.json", "w") as f:
            json_compatible_stats = {
                'transition_counts': analyzer.transition_counts,
                'transition_thresholds': analyzer.transition_thresholds,
                'natural_transitions': analyzer.natural_transition_count,
                'forced_transitions': analyzer.forced_transition_count,
                'transition_timestamps': analyzer.transition_timestamps,
                'distraction_buildup_rates': analyzer.distraction_buildup_rates,
                'average_activations_at_transition': {
                    state: np.mean(acts, axis=0).tolist() if len(acts) > 0 else [0] * len(self.thoughtseeds)
                    for state, acts in analyzer.transition_activations.items()
                }
            }
            json.dump(json_compatible_stats, f, indent=2)
        print(f"  - Transition stats saved to ./results/data/transition_stats_{self.experience_level}.json")
    
    def export_learning_history(self, state_history, activations_history, 
                               meta_awareness_history, dominant_ts_history):
        """Export full learning history to JSON"""
        # Convert history data to JSON-compatible format
        json_history_data = {
            'state_history': state_history,
            'meta_awareness_history': [float(ma) for ma in meta_awareness_history],
            'dominant_ts_history': dominant_ts_history,
            'timesteps': len(state_history),
            # Convert numpy arrays to lists
            'activations_history': [act.tolist() if isinstance(act, np.ndarray) else act 
                                   for act in activations_history]
        }

        with open(f"./results/data/learning_{self.experience_level}_history.json", "w") as f:
            json.dump(json_history_data, f, indent=2)
        print(f"  - Full learning history saved to ./results/data/learning_{self.experience_level}_history.json")
    
    def export_parameter_files(self, weights, analyzer, state_history,
                              activations_history, meta_awareness_history):
        """Export parameter files for simulation components"""
        # Export ThoughtseedNetwork parameters
        self._export_thoughtseed_parameters(activations_history, state_history)
        
        # Export MetaCognition parameters
        self._export_metacognition_parameters(meta_awareness_history, state_history)
        
        # Export MeditationStateManager parameters
        self._export_state_parameters(analyzer, state_history)
        
        print(f"  - JSON parameter files saved to ./results/data/ directory")
    
    def _export_thoughtseed_parameters(self, activations_history, state_history):
        """Export parameters for ThoughtseedNetwork"""
        thoughtseed_params = {
            "interactions": THOUGHTSEED_INTERACTIONS,
            "agent_parameters": {
                ts: {
                    "base_activation": float(np.mean([act[i] for act in activations_history])),
                    "responsiveness": float(max(0.5, 1.0 - np.std([act[i] for act in activations_history]))),
                    "decay_rate": THOUGHTSEED_AGENTS[ts]["decay_rate"],
                    "recovery_rate": THOUGHTSEED_AGENTS[ts]["recovery_rate"]
                } for i, ts in enumerate(self.thoughtseeds)
            },
            "activation_means_by_state": {
                state: {
                    ts: float(np.mean([
                        activations_history[j][i] 
                        for j, s in enumerate(state_history) if s == state
                    ])) for i, ts in enumerate(self.thoughtseeds)
                } for state in self.states if any(s == state for s in state_history)
            }
        }
        
        with open(f"./results/data/thoughtseed_params_{self.experience_level}.json", "w") as f:
            json.dump(thoughtseed_params, f, indent=2)
    
    def _export_metacognition_parameters(self, meta_awareness_history, state_history):
        """Export parameters for MetaCognitiveMonitor"""
        meta_params = {
            "meta_awareness_base": 0.8 if self.experience_level == 'expert' else 0.7,
            "meta_awareness_noise": 0.03 if self.experience_level == 'expert' else 0.05,
            "habituation_recovery": 0.5 if self.experience_level == 'expert' else 0.3,
            "average_meta_awareness_by_state": {
                state: float(np.mean([
                    meta_awareness_history[j] 
                    for j, s in enumerate(state_history) if s == state
                ])) for state in self.states if any(s == state for s in state_history)
            }
        }
        
        with open(f"./results/data/metacognition_params_{self.experience_level}.json", "w") as f:
            json.dump(meta_params, f, indent=2)
    
    def _export_state_parameters(self, analyzer, state_history):
        """Export parameters for MeditationStateManager"""
        # Convert transition counts to probabilities
        transition_probs = {}
        for source in self.states:
            total = sum(analyzer.transition_counts[source].values())
            if total > 0:
                transition_probs[source] = {
                    target: analyzer.transition_counts[source][target] / total 
                    for target in self.states
                }
            else:
                transition_probs[source] = {target: 0.0 for target in self.states}
        
        # Calculate state durations
        state_duration_stats = {
            state: {
                "mean_duration": float(np.mean([
                    sum(1 for _ in g) for k, g in itertools.groupby(state_history) 
                    if k == state
                ]) if any(s == state for s in state_history) else 0),
                "std_duration": float(np.std([
                    sum(1 for _ in g) for k, g in itertools.groupby(state_history) 
                    if k == state
                ]) if any(s == state for s in state_history) else 0)
            } for state in self.states
        }
        
        from config.meditation_config import STATE_DWELL_TIMES
        
        state_params = {
            "dwell_times": {
                state: list(STATE_DWELL_TIMES[self.experience_level][state])
                for state in self.states
            },
            "transition_probabilities": transition_probs,
            "state_duration_stats": state_duration_stats
        }
        
        with open(f"./results/data/state_params_{self.experience_level}.json", "w") as f:
            json.dump(state_params, f, indent=2)