"""
Visualization utilities for airport schedules and operations.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import numpy as np
from typing import List, Dict, Optional

from src.mdp.components.flight import Flight


def plot_runway_schedule(
    flights: List[Flight], 
    num_runways: int, 
    title: str = "Runway Schedule",
    block_duration: int = 5
):
    """
    Visualize the runway schedule as a Gantt chart.
    
    Args:
        flights: List of Flight objects
        num_runways: Number of runways
        title: Plot title
        block_duration: Duration to display for each flight block (minutes)
    """
    if not flights:
        print("No flights to visualize.")
        return

    # Setup figure
    fig, ax = plt.subplots(figsize=(15, max(4, num_runways * 1.5)))
    
    # Define colors for known aircraft types
    type_colors = {
        'regional': '#87CEEB',      # Sky Blue
        'narrow-body': '#FFA500',   # Orange
        'wide-body': '#32CD32',     # Lime Green
    }
    # Fallback color for unknown types
    default_color = '#D3D3D3'       # Light Gray
    
    # Collect all unique types for legend
    present_types = set(f.aircraft_type for f in flights)
    
    # Plot each flight
    for flight in flights:
        start_time = flight.scheduled_time
        runway_idx = flight.runway
        
        # Ensure runway index is valid for plotting
        if runway_idx >= num_runways:
            continue
            
        color = type_colors.get(flight.aircraft_type, default_color)
        
        # Determine style based on direction
        hatch = ''
        edgecolor = 'black'
        if hasattr(flight, 'direction') and flight.direction == 'departure':
            hatch = '///'
            # Make departures slightly lighter or distinct
            
        # Draw a rectangle representing the flight arrival slot
        # Center the block vertically on the runway line
        rect = patches.Rectangle(
            (start_time, runway_idx - 0.3),  # (x, y) bottom-left corner
            block_duration,                  # width
            0.6,                             # height
            linewidth=1,
            edgecolor=edgecolor,
            facecolor=color,
            hatch=hatch,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add flight ID text above the block (optional, uncomment if needed)
        # label = flight.flight_id
        # if hasattr(flight, 'direction') and flight.direction == 'departure':
        #     label += " (D)"
        #
        # ax.text(
        #     start_time + block_duration/2, 
        #     runway_idx, 
        #     label, 
        #     ha='center', 
        #     va='center', 
        #     fontsize=6,
        #     rotation=90,
        #     color='black',
        #     fontweight='bold'
        # )

    # Set axis limits and labels
    min_time = min(f.scheduled_time for f in flights)
    max_time = max(f.scheduled_time for f in flights)
    padding = 30
    
    ax.set_xlim(min_time - padding, max_time + padding)
    ax.set_ylim(-0.5, num_runways - 0.5)
    
    # Y-axis ticks for runways
    ax.set_yticks(range(num_runways))
    ax.set_yticklabels([f"Runway {i}" for i in range(num_runways)])
    
    # Format x-axis as HH:MM
    def minutes_to_time(x, pos):
        hours = int(x // 60) % 24  # Wrap around 24h
        minutes = int(x % 60)
        return f"{hours:02d}:{minutes:02d}"
        
    ax.xaxis.set_major_formatter(FuncFormatter(minutes_to_time))
    ax.set_xlabel("Time (HH:MM)")

    ax.set_title(title)
    
    # Create legend
    legend_elements = []
    # Type colors
    for atype in sorted(present_types):
        color = type_colors.get(atype, default_color)
        legend_elements.append(patches.Patch(facecolor=color, label=atype, edgecolor='black'))
    
    # Direction styles
    legend_elements.append(patches.Patch(facecolor='white', label='Arrival', edgecolor='black'))
    legend_elements.append(patches.Patch(facecolor='white', label='Departure', edgecolor='black', hatch='///'))
        
    ax.legend(handles=legend_elements, loc='upper right', title="Legend")
    
    # Add grid
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
