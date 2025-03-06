import numpy as np
from config.meditation_config import STATE_DWELL_TIMES, THOUGHTSEED_AGENTS, MEDITATION_STATE_THOUGHTSEED_ATTRACTORS

class ThoughtseedTrainer:
    """Core training logic for thoughtseed network"""
    
    def __init__(self, experience_level, thoughtseeds, states):
        self.experience_level = experience_level
        self.thoughtseeds = thoughtseeds
        self.states = states
        self.noise_level = 0.03 if experience_level == 'expert' else 0.05
        
        # Dwell times with mean ± 2 SD for variability
        self.min_max_dwell_times = {
            state: self._calculate_dwell_variability(min_time, max_time)
            for state, (min_time, max_time) in STATE_DWELL_TIMES[experience_level].items()
        }
    
    def _calculate_dwell_variability(self, min_time, max_time):
        """Calculate mean ± 2 standard deviations for dwell time variability"""
        mean_dwell = (min_time + max_time) / 2
        std_dwell = (max_time - min_time) / 6  # Assuming uniform distribution
        min_dwell = max(1, int(mean_dwell - 2 * std_dwell))
        max_dwell = int(mean_dwell + 2 * std_dwell)
        return (min_dwell, max_dwell)

    def _get_mean_dwell(self, state):
        """Calculate mean dwell time for a given state and experience level"""
        min_dwell, max_dwell = STATE_DWELL_TIMES[self.experience_level][state]
        return (min_dwell + max_dwell) / 2

    def get_target_activations(self, state, meta_awareness):
        """Generate target activations based on state, weights, and interactions"""
        # Create method with code from original get_target_activations
        # This is the core training algorithm
        state_idx = self.states.index(state)
        target_activations = np.zeros(len(self.thoughtseeds))
        attrs = MEDITATION_STATE_THOUGHTSEED_ATTRACTORS[state]

        # Adjust for state-specific dominance with attractors
        for ts_idx, ts in enumerate(self.thoughtseeds):
            if ts in attrs["primary"]:
                target_activations[ts_idx] = np.random.uniform(0.45, 0.55) * (1 + meta_awareness * 0.1)
            elif ts in attrs["secondary"]:
                target_activations[ts_idx] = np.random.uniform(0.25, 0.35) * (1 + meta_awareness * 0.1)
            else:
                target_activations[ts_idx] = np.random.uniform(0.05, 0.15) * (1 + meta_awareness * 0.1)

        # Add noise for biological variability
        target_activations += np.random.normal(0, self.noise_level, size=len(self.thoughtseeds))
        target_activations = np.clip(target_activations, 0.05, 1.0)
        return target_activations

    def get_dwell_time(self, state):
        """Generate a random dwell time within mean ± 2 SD for biological variability"""
        min_dwell, max_dwell = self.min_max_dwell_times[state]
        
        # Get the configured range from STATE_DWELL_TIMES
        config_min, config_max = STATE_DWELL_TIMES[self.experience_level][state]
        
        # Ensure minimal biological plausibility while respecting configured values
        if state in ['meta_awareness', 'redirect_breath']:
            # For brief states: at least 1 timestep, respect configured max
            min_biological = 1
            max_biological = config_max
        else:
            # For longer states: at least 3 timesteps, respect configured max
            min_biological = 3
            max_biological = config_max
        
        # Generate dwell time with proper constraints
        return max(min_biological, min(max_biological, 
                                      int(np.random.uniform(min_dwell, max_dwell))))

    def get_meta_awareness(self, state, activations):
        """Calculate meta-awareness based on state, dominant thoughtseed, and noise"""
        dominant_ts = self.thoughtseeds[np.argmax(activations)]
        base_awareness = 0.6  # Baseline for minimal self-monitoring
        noise = np.random.normal(0, self.noise_level / 2)  # Reduced noise for stability
        if state == "mind_wandering":
            if dominant_ts in ["pending_tasks", "pain_discomfort"]:
                return max(0.55, base_awareness - 0.05 + noise)
            return max(0.6, base_awareness + noise)
        elif state == "meta_awareness":
            return min(1.0 if self.experience_level == 'expert' else 0.9, base_awareness + 0.4 + noise)
        elif state == "redirect_breath":
            return min(0.85, base_awareness + 0.25 + noise)
        else:  # breath_control
            return min(0.8 if self.experience_level == 'expert' else 0.75, base_awareness + 0.2 + noise)

    def update_activations(self, current_state, activations, prev_activations, meta_awareness, current_dwell):
        """Update thoughtseed activations for a training cycle"""
        # Core activation update logic from the original train() method
        target_activations = self.get_target_activations(current_state, meta_awareness)

        # Smooth transition over 3 timesteps for biological plausibility
        if current_dwell < 3:
            alpha = (current_dwell + 1) / 3
            activations = (1 - alpha) * prev_activations + alpha * target_activations * 0.9 + prev_activations * 0.1
        else:
            activations = target_activations * 0.9 + prev_activations * 0.1  # 90% target, 10% current for momentum

        # Apply meta-awareness scaling for state-specific dominance
        if current_state == "mind_wandering" and meta_awareness < 0.6:
            for i, ts in enumerate(self.thoughtseeds):
                if ts == "breath_focus":
                    activations[i] *= 0.05  # Strongly suppress breath_focus during low meta-awareness
                elif ts in ["pain_discomfort", "pending_tasks"]:
                    activations[i] *= 1.2  # Moderate boost for distractions
                else:
                    activations[i] *= 0.5  # Increase suppression of others for balance
        # Add other special cases from the original code
        
        # Keep activations in biological range
        activations = np.clip(activations, 0.05, 1.0)
        
        return activations