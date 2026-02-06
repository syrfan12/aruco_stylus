"""
3D Trajectory Visualization for Marker Tracking
Simple, scientific, and easy-to-understand 3D plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os


def load_data(csv_path):
    """Load data from CSV file."""
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def compute_statistics(x, y, z, label):
    """Compute and return trajectory statistics."""
    displacement = np.sqrt(
        (x[-1] - x[0])**2 + 
        (y[-1] - y[0])**2 + 
        (z[-1] - z[0])**2
    )
    
    path_length = np.sum(np.sqrt(
        np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2
    ))
    
    return {
        'label': label,
        'frames': len(x),
        'displacement': displacement,
        'path_length': path_length,
        'x_range': (x.min(), x.max()),
        'y_range': (y.min(), y.max()),
        'z_range': (z.min(), z.max()),
    }


def plot_single_trajectory(ax, x, y, z, color, label, linewidth=2):
    """Plot a single trajectory on the axis."""
    ax.plot(x, y, z, color=color, linewidth=linewidth, label=label, alpha=0.9)
    # Mark start and end
    ax.scatter(x[0], y[0], z[0], c=color, s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
    ax.scatter(x[-1], y[-1], z[-1], c=color, s=100, marker='^', edgecolors='black', linewidths=1.5, zorder=5)


def plot_combined_3d(single_csv, dodeca_csv):
    """Plot 3D trajectories in a clean, scientific manner."""
    
    # Load data
    df_single = load_data(single_csv)
    df_dodeca = load_data(dodeca_csv)
    
    # Extract trajectories
    trajectories = []
    
    if df_single is not None:
        trajectories.append({
            'data': df_single,
            'marker': ('marker_x', 'marker_y', 'marker_z'),
            'tip': ('tip_x', 'tip_y', 'tip_z'),
            'marker_color': '#1f77b4',  # Blue
            'tip_color': '#ff7f0e',      # Orange
            'marker_label': 'Single Marker',
            'tip_label': 'Single Pen Tip',
            'csv_path': single_csv
        })
    
    if df_dodeca is not None:
        trajectories.append({
            'data': df_dodeca,
            'marker': ('marker_x', 'marker_y', 'marker_z'),
            'tip': ('tip_x', 'tip_y', 'tip_z'),
            'marker_color': '#2ca02c',  # Green
            'tip_color': '#d62728',      # Red
            'marker_label': 'Dodeca Marker',
            'tip_label': 'Dodeca Pen Tip',
            'csv_path': dodeca_csv
        })
    
    if not trajectories:
        print("Error: No data files found")
        return
    
    # Create 3D plot
    fig = plt.figure(figsize=(20, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    stats = []
    
    # Plot trajectories
    for traj in trajectories:
        df = traj['data']
        
        # Marker trajectory
        marker_x = df[traj['marker'][0]].values
        marker_y = df[traj['marker'][1]].values
        marker_z = df[traj['marker'][2]].values
        
        plot_single_trajectory(ax, marker_x, marker_y, marker_z, 
                              traj['marker_color'], traj['marker_label'])
        stats.append(compute_statistics(marker_x, marker_y, marker_z, traj['marker_label']))
        
        # Tip trajectory
        tip_x = df[traj['tip'][0]].values
        tip_y = df[traj['tip'][1]].values
        tip_z = df[traj['tip'][2]].values
        
        plot_single_trajectory(ax, tip_x, tip_y, tip_z, 
                              traj['tip_color'], traj['tip_label'], linewidth=1.5)
        stats.append(compute_statistics(tip_x, tip_y, tip_z, traj['tip_label']))
    
    # Configure plot
    ax.set_xlabel('X (mm)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=16, fontweight='bold')
    ax.set_zlabel('Z (mm)', fontsize=16, fontweight='bold')
    ax.set_title('Tip Trajectories Calibration', fontsize=18, fontweight='bold', pad=20)
    
    # Clean grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Create detailed legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', linewidth=2.5, label='Single Marker Center'),
        Line2D([0], [0], color='#ff7f0e', linewidth=2.5, label='Single Marker Ultrasound Probe Tip'),
        Line2D([0], [0], color='#2ca02c', linewidth=2.5, label='Multi Marker Center'),
        Line2D([0], [0], color='#d62728', linewidth=2.5, label='Multi Ultrasound Probe Tip'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
               markeredgecolor='black', markeredgewidth=1.5, label='Start Point'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, 
               markeredgecolor='black', markeredgewidth=1.5, label='End Point'),
    ]
    
    legend1 = ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5),
                       fontsize=15, framealpha=0.95, edgecolor='black', title='Legend', title_fontsize=16)
    legend1.get_frame().set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.dirname(single_csv) if single_csv else os.path.dirname(dodeca_csv)
    output_path = os.path.join(output_dir, '3d_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")
    
    # Print statistics
    print_statistics(stats)
    
    plt.show()


def print_statistics(stats):
    """Print trajectory statistics in a clean format."""
    print(f"\n{'='*80}")
    print("TRAJECTORY STATISTICS")
    print(f"{'='*80}\n")
    
    for s in stats:
        print(f"┌─ {s['label']} ({s['frames']} frames)")
        print(f"├─ Displacement (straight line):  {s['displacement']:>10.2f} mm")
        print(f"├─ Path length (actual travel):   {s['path_length']:>10.2f} mm")
        print(f"├─ X range: {s['x_range'][0]:>7.1f} ─► {s['x_range'][1]:>7.1f} mm  (Δ = {s['x_range'][1]-s['x_range'][0]:>7.1f} mm)")
        print(f"├─ Y range: {s['y_range'][0]:>7.1f} ─► {s['y_range'][1]:>7.1f} mm  (Δ = {s['y_range'][1]-s['y_range'][0]:>7.1f} mm)")
        print(f"└─ Z range: {s['z_range'][0]:>7.1f} ─► {s['z_range'][1]:>7.1f} mm  (Δ = {s['z_range'][1]-s['z_range'][0]:>7.1f} mm)")
        print()
    
    print(f"{'='*80}")
    print("\nNOTA:")
    print("  • Displacement = Linear distance from start to end point")
    print("  • Path length  = Total actual distance traveled (always ≥ displacement)")
    print("  • Range (Δ)    = Minimum to maximum coordinate difference")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Simple 3D trajectory visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_single_3d.py -l                    (use latest session)
  python plot_single_3d.py -s dataMarker/for_calib/session_XXX
  python plot_single_3d.py -f path/to/data_single.csv
        """)
    parser.add_argument('-s', '--session', type=str, help='Session directory path')
    parser.add_argument('-l', '--latest', action='store_true', help='Use latest session')
    parser.add_argument('-f', '--file', type=str, help='Direct path to CSV file')
    
    args = parser.parse_args()
    
    # Determine CSV paths
    single_csv = None
    dodeca_csv = None
    
    if args.file:
        if 'single' in args.file:
            single_csv = args.file
            dodeca_csv = args.file.replace('data_single.csv', 'data_dodeca.csv')
        elif 'dodeca' in args.file:
            dodeca_csv = args.file
            single_csv = args.file.replace('data_dodeca.csv', 'data_single.csv')
    elif args.latest:
        base_dir = 'dataMarker/for_calib'
        if os.path.exists(base_dir):
            sessions = sorted([d for d in os.listdir(base_dir) if d.startswith('session_')])
            if sessions:
                latest_session = sessions[-1]
                session_path = os.path.join(base_dir, latest_session)
                single_csv = os.path.join(session_path, 'data_single.csv')
                dodeca_csv = os.path.join(session_path, 'data_dodeca.csv')
    elif args.session:
        single_csv = os.path.join(args.session, 'data_single.csv')
        dodeca_csv = os.path.join(args.session, 'data_dodeca.csv')
    
    if not single_csv or not dodeca_csv:
        print("Error: Could not determine CSV paths. Use -h for help.")
        return
    
    print("\n" + "="*70)
    print("3D TRAJECTORY VISUALIZATION")
    print("="*70)
    plot_combined_3d(single_csv, dodeca_csv)


if __name__ == '__main__':
    main()
