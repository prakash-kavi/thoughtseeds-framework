import numpy as np

class TransitionAnalyzer:
    """Analyzes and records state transitions during training"""
    
    def __init__(self, thoughtseeds, states, experience_level):
        self.thoughtseeds = thoughtseeds
        self.states = states
        self.experience_level = experience_level
        
        # Initialize tracking structures
        self.transition_counts = {state: {next_state: 0 for next_state in states} 
                                 for state in states}
        self.transition_activations = {state: [] for state in states}
        self.natural_transition_count = 0
        self.forced_transition_count = 0
        self.distraction_buildup_rates = []
        self.transition_timestamps = []
        
        # Define transition thresholds
        self.transition_thresholds = {
            'mind_wandering': 0.25,  # Threshold for distraction level
            'meta_awareness': 0.30,  # Threshold for self-reflection
            'return_focus': 0.30     # Threshold for returning to breath focus
        }
    
    def record_transition(self, from_state, to_state, activations, is_natural=False, timestamp=None):
        """Record a transition between states"""
        self.transition_counts[from_state][to_state] += 1
        self.transition_activations[from_state].append(activations.copy())
        
        if is_natural:
            self.natural_transition_count += 1
            if timestamp is not None:
                self.transition_timestamps.append(timestamp)
        else:
            self.forced_transition_count += 1
            
    def record_distraction_buildup(self, rate):
        """Record distraction buildup rate"""
        self.distraction_buildup_rates.append(rate)
    
    def check_for_natural_transition(self, current_state, activations):
        """Check if conditions for natural transition are met"""
        # From original code's natural transition logic
        next_state = None
        should_transition = False
        
        # Calculate distraction level from pain and pending tasks
        distraction_level = activations[self.thoughtseeds.index("pain_discomfort")] + \
                            activations[self.thoughtseeds.index("pending_tasks")]
        
        # Check for transition conditions based on activation patterns
        if current_state in ["breath_control", "redirect_breath"]:
            # High distraction can lead to mind wandering
            if distraction_level > self.transition_thresholds['mind_wandering']:
                # EXPERT ADJUSTMENT: Experts are better at resisting distractions
                if self.experience_level == 'expert':
                    # Experts have 60% chance to resist the distraction
                    if np.random.random() < 0.6:
                        # Resist distraction - don't transition
                        return False, None
                
                next_state = "mind_wandering"
                should_transition = True
                
        elif current_state == "mind_wandering":
            # High self-reflection can lead to meta-awareness
            if activations[self.thoughtseeds.index("self_reflection")] > self.transition_thresholds['meta_awareness']:
                # EXPERT ADJUSTMENT: Experts are better at noticing mind-wandering
                if self.experience_level == 'expert':
                    # Boost transition probability by 30%
                    should_transition = True
                    next_state = "meta_awareness"
                else:
                    next_state = "meta_awareness"
                    should_transition = True
        
        elif current_state == "meta_awareness":
            # High breath focus can return to breath control
            # High equanimity can lead to redirect breath
            if activations[self.thoughtseeds.index("breath_focus")] > self.transition_thresholds['return_focus']:
                next_state = "breath_control"
                should_transition = True
            elif activations[self.thoughtseeds.index("equanimity")] > self.transition_thresholds['return_focus']:
                next_state = "redirect_breath"
                should_transition = True
                
        return should_transition, next_state