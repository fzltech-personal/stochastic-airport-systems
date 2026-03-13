"""
Visualization utilities for airport schedules and operations.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Optional, Union

from src.mdp.components.flight import ScheduledFlight
from src.simulation.realization import ActiveFlight


def plot_runway_schedule(
    flights: List[Union[ScheduledFlight, ActiveFlight]],
    num_runways: int,
    title: str = "Runway Schedule",
    block_duration: int = 5
):
    """
    Visualize the runway schedule as a Gantt chart.

    Args:
        flights: List of ScheduledFlight or ActiveFlight objects
        num_runways: Number of runways
        title: Plot title
        block_duration: Duration to display for each flight block (minutes)
    """
    if not flights:
        print("No flights to visualize.")
        return

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define Y-axis (Runways)
    ax.set_ylim(-1, num_runways)
    ax.set_yticks(range(num_runways))
    ax.set_yticklabels([f"Runway {i}" for i in range(num_runways)])
    ax.set_ylabel("Runway Index")

    # Define X-axis (Time)
    times = []
    for f in flights:
        if isinstance(f, ActiveFlight):
            times.append(f.actual_arrival_time)
        else:
            times.append(f.scheduled_time)
            
    if not times:
        return

    min_time = min(times)
    max_time = max(times)
    ax.set_xlim(min_time - 30, max_time + 30)
    ax.set_xlabel("Time (minutes)")

    # Colors for different aircraft types
    colors = {
        "narrow_body": "blue",
        "wide_body": "red",
        "regional": "green"
    }
    default_color = "gray"

    # Plot flights
    for f in flights:
        # Determine time, runway, type, and direction
        if isinstance(f, ActiveFlight):
            t = f.actual_arrival_time
            # ActiveFlight uses schedule properties for fixed attributes
            r = f.schedule.runway
            atype = f.schedule.aircraft_type
            direction = f.schedule.direction
        else:
            t = f.scheduled_time
            r = f.runway
            atype = f.aircraft_type
            direction = f.direction

        # Choose color
        color = colors.get(atype, default_color)
        
        # Visual distinction for departures (hatched)
        hatch = "///" if direction == "departure" else None

        # Draw rectangle
        rect = patches.Rectangle(
            (t, r - 0.4),       # (x, y)
            block_duration,     # width
            0.8,                # height
            facecolor=color,
            edgecolor="black",
            alpha=0.6,
            hatch=hatch
        )
        ax.add_patch(rect)

    ax.set_title(title)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
