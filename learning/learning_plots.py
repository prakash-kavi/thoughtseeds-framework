import matplotlib.pyplot as plt
import numpy as np

def plot_results(learner):
    """Plot meditation states, thoughtseed activations, and meta-awareness"""
    # Get data from learner
    time_steps = np.arange(learner.timesteps)
    experience_level = learner.experience_level
    thoughtseeds = learner.thoughtseeds
    states = learner.states
    
    # Create state_indices for plotting
    state_indices = {state: i for i, state in enumerate(states)}
    
    # Improved color palette
    thoughtseed_colors = {
        'self_reflection': '#4363d8',      # Blue
        'breath_focus': '#f58231',         # Orange
        'equanimity': '#3cb44b',           # Green
        'pain_discomfort': '#e6194B',      # Red
        'pending_tasks': '#911eb4'         # Purple
    }
    
    # State colors for better visualization
    state_colors = {
        "breath_control": "#7fcdbb",
        "mind_wandering": "#2c7fb8",
        "meta_awareness": "#f0ad4e", 
        "redirect_breath": "#5ab4ac"
    }

    # Plot meditation states with improved styling
    plt.figure(figsize=(12, 3))
    state_indices_list = [state_indices[state] for state in learner.state_history]
    
    # Plot state transitions as step function with filled areas for each state
    plt.step(time_steps, state_indices_list, where='post', color='#444444', linewidth=1.5, alpha=0.7)
    
    for i, state in enumerate(states):
        mask = np.array(state_indices_list) == i
        if any(mask):
            plt.fill_between(time_steps, i, i+1, where=mask, step='post',
                           color=state_colors[state], alpha=0.3)
    
    plt.xlabel('Timestep', fontsize=11)
    plt.ylabel('State', fontsize=11)
    plt.yticks([0, 1, 2, 3], states)
    plt.title(f'Learning: Meditation States Over Time ({experience_level.capitalize()})', 
             fontsize=13, fontweight='bold')
    plt.xlim(0, learner.timesteps)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'./results/plots/training/learning_{experience_level}_meditation_states.png', 
               dpi=300, bbox_inches='tight')
    plt.close()

    # Plot thoughtseed activations and meta-awareness with consistent ranges - SWAPPED ORDER
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Meta-awareness on top (ax1) with smoothing
    smoothed_meta = np.zeros_like(learner.meta_awareness_history)
    alpha = 0.3
    smoothed_meta[0] = learner.meta_awareness_history[0]
    for j in range(1, len(learner.meta_awareness_history)):
        smoothed_meta[j] = (1 - alpha) * smoothed_meta[j-1] + alpha * learner.meta_awareness_history[j]
    
    ax1.plot(time_steps, smoothed_meta, color=thoughtseed_colors['self_reflection'], linewidth=2.5, label='Meta-Awareness')
    ax1.fill_between(time_steps, smoothed_meta, alpha=0.2, color=thoughtseed_colors['self_reflection'])
    ax1.set_ylabel('Meta-Awareness', fontsize=11)
    ax1.set_title(f'Learning: Meta-Awareness ({experience_level.capitalize()})', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9, fancybox=True, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0.55, 1.0)  # Fixed y-axis for meta-awareness (0.55–1.0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Thoughtseed activations below (ax2)
    reordered_thoughtseeds = ['self_reflection', 'breath_focus', 'equanimity', 'pain_discomfort', 'pending_tasks']
    for ts in reordered_thoughtseeds:
        i = thoughtseeds.index(ts)
        activations = [act[i] for act in learner.activations_history]
        smoothed_activations = np.zeros_like(activations)
        alpha = 0.3  # Smoothing for neural inertia
        smoothed_activations[0] = activations[0]
        for j in range(1, len(activations)):
            smoothed_activations[j] = (1 - alpha) * smoothed_activations[j-1] + alpha * activations[j]
        ax2.plot(time_steps, smoothed_activations, label=ts, color=thoughtseed_colors[ts], linewidth=2)
    
    ax2.set_xlabel('Timestep', fontsize=11)
    ax2.set_ylabel('Activation', fontsize=11)
    ax2.set_title(f'Learning: Thoughtseed Activations ({experience_level.capitalize()})', 
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9, fancybox=True, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(0.0, 1.05)  # Fixed y-axis for activations
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(0, learner.timesteps)
    
    plt.show()

    # plt.tight_layout()
    # plt.savefig(f'./results/plots/training/learning_{experience_level}_thoughtseed_meta_activations.png', 
    #            dpi=300, bbox_inches='tight')
    # plt.close()
    
def plot_side_by_side_transition_matrices(learner_novice, learner_expert):
    """Plot transition matrices side by side for comparison"""
    # Get states and setup figure
    states = learner_novice.states
    
    # Create figure with proper spacing for colorbar
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Process both models
    learners = [learner_novice, learner_expert] 
    titles = ['Novice: State Transition Probabilities', 'Expert: State Transition Probabilities']
    
    # Setup for shared color scale
    all_probs = []
    
    # First pass to get all probabilities for consistent color scale
    for learner in learners:
        transition_matrix = np.zeros((len(learner.states), len(learner.states)))
        for i, source in enumerate(learner.states):
            for j, target in enumerate(learner.states):
                transition_matrix[i, j] = learner.analyzer.transition_counts[source][target]
        
        # Normalize by row
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        norm_matrix = np.zeros_like(transition_matrix)
        for i in range(len(learner.states)):
            if row_sums[i] > 0:
                norm_matrix[i] = transition_matrix[i] / row_sums[i]
        
        all_probs.extend(norm_matrix.flatten())
    
    # Get max for consistent color scale
    vmax = max(all_probs) if all_probs else 1.0
    
    # Create both plots
    for idx, (learner, ax, title) in enumerate(zip(learners, axes, titles)):
        transition_matrix = np.zeros((len(learner.states), len(learner.states)))
        for i, source in enumerate(learner.states):
            for j, target in enumerate(learner.states):
                transition_matrix[i, j] = learner.analyzer.transition_counts[source][target]
        
        # Normalize by row
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        norm_matrix = np.zeros_like(transition_matrix)
        for i in range(len(learner.states)):
            if row_sums[i] > 0:
                norm_matrix[i] = transition_matrix[i] / row_sums[i]
        
        im = ax.imshow(norm_matrix, cmap='viridis', interpolation='nearest', vmax=vmax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(np.arange(len(learner.states)))
        ax.set_yticks(np.arange(len(learner.states)))
        ax.set_xticklabels(learner.states, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(learner.states)
        
        # Add text annotations
        for i in range(len(learner.states)):
            for j in range(len(learner.states)):
                if norm_matrix[i, j] > 0:
                    ax.text(j, i, f'{norm_matrix[i, j]:.2f}',
                          ha='center', va='center', 
                          color='white' if norm_matrix[i, j] > 0.4 else 'black')
    
    # Add colorbar to the right of the second plot
    # Leave space for the colorbar
    fig.subplots_adjust(right=0.85)
    
    # Create a new axis for the colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    
    # Add colorbar
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Transition Probability', rotation=270, labelpad=20)
              
    plt.savefig('./results/plots/training/transition_matrices_comparison.png', 
               dpi=300, bbox_inches='tight')
    plt.close()