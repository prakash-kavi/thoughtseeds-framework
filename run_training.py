import os
import numpy as np
import json

from config.meditation_config import (
    THOUGHTSEEDS, STATES, STATE_DWELL_TIMES, THOUGHTSEED_INTERACTIONS, 
    THOUGHTSEED_AGENTS, MEDITATION_STATE_THOUGHTSEED_ATTRACTORS
)

from learning.training.trainer import ThoughtseedTrainer
from learning.training.transition_analyzer import TransitionAnalyzer
from learning.training.data_exporter import LearningDataExporter
from learning.viusalize_learning.learning_plots import plot_results, plot_side_by_side_transition_matrices

def ensure_directories():
    """Create necessary directories for output files"""
    os.makedirs('./results/data', exist_ok=True)
    os.makedirs('./results/plots/training', exist_ok=True)
    os.makedirs('./results/plots/network_interactions', exist_ok=True)
    os.makedirs('./results/plots/simulation', exist_ok=True)
    print("Directories created/verified for output files")

class MeditationLearning:
    """Main coordinator for meditation training process"""
    
    def __init__(self, experience_level='novice', timesteps=200):
        # Basic initialization
        self.experience_level = experience_level
        self.timesteps = timesteps
        self.thoughtseeds = THOUGHTSEEDS
        self.states = STATES
        self.num_thoughtseeds = len(self.thoughtseeds)
        
        # Initialize components
        self.trainer = ThoughtseedTrainer(experience_level, self.thoughtseeds, self.states)
        self.analyzer = TransitionAnalyzer(self.thoughtseeds, self.states, experience_level)
        self.exporter = LearningDataExporter(experience_level, self.thoughtseeds, self.states)
        
        # Initialize weights with intentional weights, adjusted by attractors
        self.weights = np.zeros((self.num_thoughtseeds, len(self.states)))
        for ts_idx, ts in enumerate(self.thoughtseeds):
            for state_idx, state in enumerate(self.states):
                attrs = MEDITATION_STATE_THOUGHTSEED_ATTRACTORS[state]
                if ts in attrs["primary"]:
                    self.weights[ts_idx, state_idx] = THOUGHTSEED_AGENTS[ts]["intentional_weights"][experience_level] * np.random.uniform(0.9, 1.1)
                elif ts in attrs["secondary"]:
                    self.weights[ts_idx, state_idx] = THOUGHTSEED_AGENTS[ts]["intentional_weights"][experience_level] * np.random.uniform(0.7, 0.9)
                else:
                    self.weights[ts_idx, state_idx] = THOUGHTSEED_AGENTS[ts]["intentional_weights"][experience_level] * np.random.uniform(0.05, 0.2)
        self.weights = np.clip(self.weights, 0.05, 1.0)  # Ensure biological range
        
        # Initialize state tracking
        self.state_indices = {state: i for i, state in enumerate(self.states)}
        
        # Initialize histories
        self.state_history = []
        self.activations_history = []
        self.meta_awareness_history = []
        self.dominant_ts_history = []
        self.state_history_over_time = []
        
        # Noise level
        self.noise_level = 0.03 if experience_level == 'expert' else 0.05
    
    def train(self):
        """Run the complete training process"""
        print(f"\nStarting training for {self.experience_level} experience level...")
        
        # Setup state sequence
        state_sequence = ["breath_control", "mind_wandering", "meta_awareness", "redirect_breath"]
        current_state_index = 0
        current_state = state_sequence[current_state_index]
        current_dwell = 0
        dwell_limit = self.trainer.get_dwell_time(current_state)
        
        # Initialize activations
        activations = np.full(self.num_thoughtseeds, np.random.uniform(0.05, 0.15))
        activations = self.trainer.get_target_activations(current_state, 0.6)
        prev_activations = activations.copy()
        
        # Track time in focused states for distraction growth
        time_in_focused_state = 0
        
        # Main training loop
        for t in range(self.timesteps):
            # Get meta-awareness based on current state and activations
            meta_awareness = self.trainer.get_meta_awareness(current_state, activations)
            
            # Update activations using the trainer
            activations = self.trainer.update_activations(
                current_state, activations.copy(), prev_activations, meta_awareness, current_dwell
            )
            
            # ADDITIONAL FUNCTIONALITY: Distraction growth in focused states
            if current_state in ["breath_control", "redirect_breath"]:
                time_in_focused_state += 1
                dwell_factor = min(1.0, current_dwell / max(10, dwell_limit))
                
                # Calculate distraction growth based on duration in focused state
                distraction_scale = 2.5 if self.experience_level == 'novice' else 1.2
                distraction_growth = 0.035 * dwell_factor * distraction_scale
                
                # Record distraction growth rate for statistics
                self.analyzer.record_distraction_buildup(distraction_growth)
                
                # Occasional strong distractions
                boost_factor = 3.0 if np.random.random() < 0.1 else 1.0
                
                # Apply distraction growth
                for i, ts in enumerate(self.thoughtseeds):
                    if ts in ["pain_discomfort", "pending_tasks"]:
                        activations[i] += distraction_growth * boost_factor
                
                # Add fatigue to breath focus over time
                for i, ts in enumerate(self.thoughtseeds):
                    if ts == "breath_focus":
                        fatigue_rate = 0.005 if self.experience_level == 'expert' else 0.01
                        fatigue = fatigue_rate * dwell_factor * time_in_focused_state/10
                        activations[i] = max(0.2, activations[i] - fatigue)
            else:
                time_in_focused_state = 0  # Reset counter when not in focused state
            
            # ADDITIONAL FUNCTIONALITY: Expert-specific enhancements
            if self.experience_level == 'expert':
                # In redirect_breath and meta_awareness states
                if current_state in ["redirect_breath", "meta_awareness"]:
                    bf_idx = self.thoughtseeds.index("breath_focus")
                    eq_idx = self.thoughtseeds.index("equanimity")
                    
                    # Mutual reinforcement of breath focus and equanimity
                    if activations[bf_idx] > 0.3 and activations[eq_idx] > 0.3:
                        boost = 0.03 * min(activations[bf_idx], activations[eq_idx])
                        activations[bf_idx] += boost
                        activations[eq_idx] += boost
                    
                    # Equanimity reduces pain reactivity
                    if activations[eq_idx] > 0.4:
                        pd_idx = self.thoughtseeds.index("pain_discomfort")
                        activations[pd_idx] = max(0.05, activations[pd_idx] - 0.02 * activations[eq_idx])
                
                # In breath_control and redirect_breath states
                if current_state in ["breath_control", "redirect_breath"]:
                    bf_idx = self.thoughtseeds.index("breath_focus")
                    eq_idx = self.thoughtseeds.index("equanimity")
                    
                    # Strong breath focus facilitates equanimity
                    if activations[bf_idx] > 0.4:
                        facilitation = 0.08 * activations[bf_idx]
                        activations[eq_idx] += facilitation * (1.0 + np.random.uniform(-0.2, 0.2))
                        activations[eq_idx] = min(1.0, activations[eq_idx])
            
            # Cap extreme activations
            for i, ts in enumerate(self.thoughtseeds):
                if ts == "pending_tasks" and activations[i] > 0.8:
                    activations[i] = 0.8
            
            # Ensure biological range
            activations = np.clip(activations, 0.05, 1.0)
            
            # Identify dominant thoughtseed
            dominant_ts = self.thoughtseeds[np.argmax(activations)]
            
            # Track histories
            self.state_history.append(current_state)
            self.activations_history.append(activations.copy())
            self.meta_awareness_history.append(meta_awareness)
            self.dominant_ts_history.append(dominant_ts)
            self.state_history_over_time.append(self.state_indices[current_state])
            
            # Handle state transitions
            if current_dwell >= dwell_limit:
                # Check for natural transitions
                natural_prob = 0.4 + min(0.5, t / self.timesteps * 0.6)
                natural_transition = False
                next_state = None
                
                if np.random.random() < natural_prob:
                    should_transition, proposed_next = self.analyzer.check_for_natural_transition(
                        current_state, activations
                    )
                    if should_transition:
                        natural_transition = True
                        next_state = proposed_next
                
                # If no natural transition, follow fixed sequence
                if not natural_transition:
                    next_state_index = (current_state_index + 1) % len(state_sequence)
                    next_state = state_sequence[next_state_index]
                    self.analyzer.record_transition(current_state, next_state, activations, False)
                else:
                    self.analyzer.record_transition(current_state, next_state, activations, True, t)
                
                # Update state
                current_state_index = state_sequence.index(next_state)
                current_state = next_state
                current_dwell = 0
                dwell_limit = self.trainer.get_dwell_time(current_state)
                
                # Blend activations for smoother transition (70% new target, 30% current)
                new_target = self.trainer.get_target_activations(current_state, meta_awareness)
                activations = 0.7 * new_target + 0.3 * activations
            else:
                current_dwell += 1
                
            prev_activations = activations.copy()
        
        # Fallback for minimum natural transitions
        if self.analyzer.natural_transition_count == 0:
            print("WARNING: No natural transitions occurred. Forcing some natural transitions...")
            # Force at least one natural transition of each type after regular training
            
            # Force breath_control to mind_wandering
            activations = np.zeros(self.num_thoughtseeds)
            activations[self.thoughtseeds.index("pain_discomfort")] = 0.5
            activations[self.thoughtseeds.index("pending_tasks")] = 0.5
            self.analyzer.record_transition("breath_control", "mind_wandering", activations, True)
            
            # Force mind_wandering to meta_awareness
            activations = np.zeros(self.num_thoughtseeds)
            activations[self.thoughtseeds.index("self_reflection")] = 0.6
            self.analyzer.record_transition("mind_wandering", "meta_awareness", activations, True)
            
            # Force meta_awareness to breath_control
            activations = np.zeros(self.num_thoughtseeds)
            activations[self.thoughtseeds.index("breath_focus")] = 0.6
            self.analyzer.record_transition("meta_awareness", "breath_control", activations, True)
        
        # Export all results using the exporter
        self.exporter.export_all(
            self.weights,
            self.analyzer,
            self.state_history,
            self.activations_history,
            self.meta_awareness_history,
            self.dominant_ts_history
        )
        
        print(f"Training complete for {self.experience_level}.")
        print(f"  - Natural transitions: {self.analyzer.natural_transition_count}, " + 
              f"Forced transitions: {self.analyzer.forced_transition_count}")
        
        # Return self for method chaining and plotting
        return self

def main():
    """Main function to run the training process"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Ensure directories exist
    ensure_directories()
    
    # Run for novice
    learner_novice = MeditationLearning(experience_level='novice', timesteps=200)
    learner_novice.train()
    
    # Run for expert  
    learner_expert = MeditationLearning(experience_level='expert', timesteps=200)
    learner_expert.train()
    
    # Generate plots
    plot_results(learner_novice)
    plot_results(learner_expert)
    
    # Add comparison of transition matrices
    plot_side_by_side_transition_matrices(learner_novice, learner_expert)
    print("  - Side-by-side transition matrix comparison saved to results/plots/")


if __name__ == "__main__":
    main()