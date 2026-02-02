"""
3D Visualization for Single Marker and Dodecahedron Tracking
- Plot single marker position (center)
- Plot dodecahedron position (center)
- Plot pen tip position
- Show both trajectories in 3D space
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os


def plot_combined_3d(single_csv, dodeca_csv, tip_csv=None):
    """Plot 3D trajectories combining single marker, dodecahedron, and both pen tips."""
    
    # Load single marker data
    if os.path.exists(single_csv):
        df_single = pd.read_csv(single_csv)
        single_x = df_single['marker_x'].values
        single_y = df_single['marker_y'].values
        single_z = df_single['marker_z'].values
        single_tip_x = df_single['tip_x'].values
        single_tip_y = df_single['tip_y'].values
        single_tip_z = df_single['tip_z'].values
        print(f"Loaded {len(df_single)} data points from single marker")
    else:
        print(f"Warning: {single_csv} not found")
        single_x = single_y = single_z = None
        single_tip_x = single_tip_y = single_tip_z = None
    
    # Load dodeca marker data
    if os.path.exists(dodeca_csv):
        df_dodeca = pd.read_csv(dodeca_csv)
        dodeca_x = df_dodeca['marker_x'].values
        dodeca_y = df_dodeca['marker_y'].values
        dodeca_z = df_dodeca['marker_z'].values
        dodeca_tip_x = df_dodeca['tip_x'].values
        dodeca_tip_y = df_dodeca['tip_y'].values
        dodeca_tip_z = df_dodeca['tip_z'].values
        print(f"Loaded {len(df_dodeca)} data points from dodeca marker")
    else:
        print(f"Warning: {dodeca_csv} not found")
        dodeca_x = dodeca_y = dodeca_z = None
        dodeca_tip_x = dodeca_tip_y = dodeca_tip_z = None
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot single marker trajectory
    if single_x is not None:
        ax.plot(single_x, single_y, single_z, 'b-', linewidth=2.5, alpha=0.8, label='Single Marker Center')
        ax.scatter(single_x[0], single_y[0], single_z[0], c='blue', s=150, marker='o', edgecolors='darkblue', linewidths=2)
        ax.scatter(single_x[-1], single_y[-1], single_z[-1], c='blue', s=150, marker='s', edgecolors='darkblue', linewidths=2)
    
    # Plot dodeca marker trajectory
    if dodeca_x is not None:
        ax.plot(dodeca_x, dodeca_y, dodeca_z, 'g-', linewidth=2.5, alpha=0.8, label='Dodeca Center')
        ax.scatter(dodeca_x[0], dodeca_y[0], dodeca_z[0], c='green', s=150, marker='o', edgecolors='darkgreen', linewidths=2)
        ax.scatter(dodeca_x[-1], dodeca_y[-1], dodeca_z[-1], c='green', s=150, marker='s', edgecolors='darkgreen', linewidths=2)
    
    # Plot single marker's pen tip trajectory
    if single_tip_x is not None:
        ax.plot(single_tip_x, single_tip_y, single_tip_z, 'r-', linewidth=2.5, alpha=0.8, label='Single Pen Tip')
        ax.scatter(single_tip_x[0], single_tip_y[0], single_tip_z[0], c='red', s=150, marker='o', edgecolors='darkred', linewidths=2)
        ax.scatter(single_tip_x[-1], single_tip_y[-1], single_tip_z[-1], c='red', s=150, marker='s', edgecolors='darkred', linewidths=2)
    
    # Plot dodeca marker's pen tip trajectory
    if dodeca_tip_x is not None:
        ax.plot(dodeca_tip_x, dodeca_tip_y, dodeca_tip_z, 'm-', linewidth=2.5, alpha=0.8, label='Dodeca Pen Tip')
        ax.scatter(dodeca_tip_x[0], dodeca_tip_y[0], dodeca_tip_z[0], c='magenta', s=150, marker='o', edgecolors='purple', linewidths=2)
        ax.scatter(dodeca_tip_x[-1], dodeca_tip_y[-1], dodeca_tip_z[-1], c='magenta', s=150, marker='s', edgecolors='purple', linewidths=2)
    
    ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
    ax.set_title('3D Trajectories: Single + Dodeca Markers & Both Pen Tips', fontweight='bold', fontsize=14)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.dirname(single_csv) if single_x is not None else os.path.dirname(dodeca_csv)
    output_path = os.path.join(output_dir, '3d_plot_combined.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] {output_path}")
    
    # Statistics
    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}")
    
    if single_x is not None:
        print(f"\nSingle Marker Center ({len(df_single)} frames):")
        print(f"  X range: {single_x.min():.2f} to {single_x.max():.2f} mm (Δ={single_x.max()-single_x.min():.2f} mm)")
        print(f"  Y range: {single_y.min():.2f} to {single_y.max():.2f} mm (Δ={single_y.max()-single_y.min():.2f} mm)")
        print(f"  Z range: {single_z.min():.2f} to {single_z.max():.2f} mm (Δ={single_z.max()-single_z.min():.2f} mm)")
        single_dist = np.sqrt(
            (single_x[-1] - single_x[0])**2 + 
            (single_y[-1] - single_y[0])**2 + 
            (single_z[-1] - single_z[0])**2
        )
        print(f"  Total displacement: {single_dist:.2f} mm")
    
    if dodeca_x is not None:
        print(f"\nDodeca Marker Center ({len(df_dodeca)} frames):")
        print(f"  X range: {dodeca_x.min():.2f} to {dodeca_x.max():.2f} mm (Δ={dodeca_x.max()-dodeca_x.min():.2f} mm)")
        print(f"  Y range: {dodeca_y.min():.2f} to {dodeca_y.max():.2f} mm (Δ={dodeca_y.max()-dodeca_y.min():.2f} mm)")
        print(f"  Z range: {dodeca_z.min():.2f} to {dodeca_z.max():.2f} mm (Δ={dodeca_z.max()-dodeca_z.min():.2f} mm)")
        dodeca_dist = np.sqrt(
            (dodeca_x[-1] - dodeca_x[0])**2 + 
            (dodeca_y[-1] - dodeca_y[0])**2 + 
            (dodeca_z[-1] - dodeca_z[0])**2
        )
        print(f"  Total displacement: {dodeca_dist:.2f} mm")
    
    if single_tip_x is not None:
        print(f"\nSingle Pen Tip Movement:")
        print(f"  X range: {single_tip_x.min():.2f} to {single_tip_x.max():.2f} mm (Δ={single_tip_x.max()-single_tip_x.min():.2f} mm)")
        print(f"  Y range: {single_tip_y.min():.2f} to {single_tip_y.max():.2f} mm (Δ={single_tip_y.max()-single_tip_y.min():.2f} mm)")
        print(f"  Z range: {single_tip_z.min():.2f} to {single_tip_z.max():.2f} mm (Δ={single_tip_z.max()-single_tip_z.min():.2f} mm)")
        tip_dist = np.sqrt(
            (single_tip_x[-1] - single_tip_x[0])**2 + 
            (single_tip_y[-1] - single_tip_y[0])**2 + 
            (single_tip_z[-1] - single_tip_z[0])**2
        )
        print(f"  Total displacement: {tip_dist:.2f} mm")
    
    if dodeca_tip_x is not None:
        print(f"\nDodeca Pen Tip Movement:")
        print(f"  X range: {dodeca_tip_x.min():.2f} to {dodeca_tip_x.max():.2f} mm (Δ={dodeca_tip_x.max()-dodeca_tip_x.min():.2f} mm)")
        print(f"  Y range: {dodeca_tip_y.min():.2f} to {dodeca_tip_y.max():.2f} mm (Δ={dodeca_tip_y.max()-dodeca_tip_y.min():.2f} mm)")
        print(f"  Z range: {dodeca_tip_z.min():.2f} to {dodeca_tip_z.max():.2f} mm (Δ={dodeca_tip_z.max()-dodeca_tip_z.min():.2f} mm)")
        tip_dist = np.sqrt(
            (dodeca_tip_x[-1] - dodeca_tip_x[0])**2 + 
            (dodeca_tip_y[-1] - dodeca_tip_y[0])**2 + 
            (dodeca_tip_z[-1] - dodeca_tip_z[0])**2
        )
        print(f"  Total displacement: {tip_dist:.2f} mm")
    
    # Compare marker centers vs their respective tips
    if single_x is not None and single_tip_x is not None:
        distances = np.sqrt(
            (single_tip_x - single_x)**2 + 
            (single_tip_y - single_y)**2 + 
            (single_tip_z - single_z)**2
        )
        print(f"\nSingle Marker-to-Tip Distance:")
        print(f"  Mean: {distances.mean():.2f} mm")
        print(f"  Std:  {distances.std():.2f} mm")
        print(f"  Min:  {distances.min():.2f} mm")
        print(f"  Max:  {distances.max():.2f} mm")
    
    if dodeca_x is not None and dodeca_tip_x is not None:
        distances = np.sqrt(
            (dodeca_tip_x - dodeca_x)**2 + 
            (dodeca_tip_y - dodeca_y)**2 + 
            (dodeca_tip_z - dodeca_z)**2
        )
        print(f"\nDodeca Marker-to-Tip Distance:")
        print(f"  Mean: {distances.mean():.2f} mm")
        print(f"  Std:  {distances.std():.2f} mm")
        print(f"  Min:  {distances.min():.2f} mm")
        print(f"  Max:  {distances.max():.2f} mm")
    
    plt.show()


def plot_dodeca_marker_3d(csv_path):
    """Deprecated - use plot_combined_3d instead"""
    pass


def main():
    parser = argparse.ArgumentParser(description='3D visualization for single and dodecahedron marker tracking (combined view)')
    parser.add_argument('-s', '--session', type=str, help='Session directory (e.g., dataMarker/for_calib/session_XXX)')
    parser.add_argument('-l', '--latest', action='store_true', help='Use latest session')
    parser.add_argument('-f', '--file', type=str, help='Direct path to data CSV file')
    
    args = parser.parse_args()
    
    # Determine CSV paths
    if args.file:
        # Direct file path provided
        if 'single' in args.file:
            single_csv = args.file
            dodeca_csv = args.file.replace('data_single.csv', 'data_dodeca.csv')
        elif 'dodeca' in args.file:
            dodeca_csv = args.file
            single_csv = args.file.replace('data_dodeca.csv', 'data_single.csv')
        else:
            single_csv = args.file
            dodeca_csv = None
    elif args.latest:
        # Use latest session
        base_dir = 'dataMarker/for_calib'
        sessions = sorted([d for d in os.listdir(base_dir) if d.startswith('session_')])
        if not sessions:
            print("Error: No sessions found")
            return
        latest_session = sessions[-1]
        session_path = os.path.join(base_dir, latest_session)
        single_csv = os.path.join(session_path, 'data_single.csv')
        dodeca_csv = os.path.join(session_path, 'data_dodeca.csv')
    elif args.session:
        # Use specified session
        single_csv = os.path.join(args.session, 'data_single.csv')
        dodeca_csv = os.path.join(args.session, 'data_dodeca.csv')
    else:
        print("Error: Specify session with -s <dir>, use -l for latest, or -f for direct file path")
        return
    
    # Plot combined 3D visualization
    print("\n" + "="*60)
    print("COMBINED 3D VISUALIZATION (Single + Dodeca + Pen Tip)")
    print("="*60)
    plot_combined_3d(single_csv, dodeca_csv)


if __name__ == '__main__':
    main()
