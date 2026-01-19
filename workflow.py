"""
Helper script to manage the two-phase recording workflow
Provides quick commands for recording and processing
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime


def record():
    """Start pure recording."""
    print("=" * 60)
    print("Phase 1: Pure Recording Mode")
    print("=" * 60)
    print("\nStarting camera recording without marker detection...")
    print("Press ESC to exit\n")
    
    try:
        subprocess.run([sys.executable, 'recorder_pure.py'], check=False)
    except KeyboardInterrupt:
        print("\nRecording interrupted")


def process(session_dir=None):
    """Start post-processing."""
    print("=" * 60)
    print("Phase 2: Post-Processing")
    print("=" * 60)
    
    if session_dir is None:
        # Find latest session
        base_dir = 'dataMarker'
        if not os.path.exists(base_dir):
            print(f"Error: {base_dir} directory not found")
            return
        
        sessions = sorted([d for d in os.listdir(base_dir) if d.startswith('session_')])
        if not sessions:
            print("Error: No sessions found")
            return
        
        session_dir = os.path.join(base_dir, sessions[-1])
    
    print(f"\nProcessing session: {session_dir}")
    
    # Check if raw_camera.mp4 exists
    video_path = os.path.join(session_dir, 'raw_camera.mp4')
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found")
        return
    
    print(f"Video size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
    print("\nStarting marker detection and analysis...\n")
    
    try:
        subprocess.run([sys.executable, 'processor_video.py', '-s', session_dir], check=False)
    except KeyboardInterrupt:
        print("\nProcessing interrupted")


def list_sessions():
    """List all available sessions."""
    base_dir = 'dataMarker'
    if not os.path.exists(base_dir):
        print(f"No {base_dir} directory found")
        return
    
    sessions = sorted([d for d in os.listdir(base_dir) if d.startswith('session_')], reverse=True)
    
    if not sessions:
        print("No sessions found")
        return
    
    print("\nAvailable Sessions:")
    print("-" * 80)
    
    for i, session in enumerate(sessions, 1):
        session_path = os.path.join(base_dir, session)
        raw_video = os.path.join(session_path, 'raw_camera.mp4')
        tracked_video = os.path.join(session_path, 'tracked.mp4')
        csv_file = os.path.join(session_path, 'data.csv')
        
        # Get file info
        raw_exists = os.path.exists(raw_video)
        tracked_exists = os.path.exists(tracked_video)
        csv_exists = os.path.exists(csv_file)
        
        status = []
        if raw_exists:
            size_mb = os.path.getsize(raw_video) / (1024*1024)
            status.append(f"raw({size_mb:.1f}MB)")
        if tracked_exists:
            status.append("tracked")
        if csv_exists:
            status.append("csv")
        
        status_str = " | ".join(status) if status else "incomplete"
        
        print(f"{i:2d}. {session}")
        print(f"    Status: {status_str}")
        print()


def quick_workflow():
    """Run complete workflow: record then process."""
    print("=" * 60)
    print("Quick Workflow: Record -> Process")
    print("=" * 60)
    print()
    
    # Step 1: Record
    print("STEP 1: Recording")
    print("-" * 60)
    input("Press Enter to start recording (ESC to exit)...")
    record()
    
    print("\nRecording completed!")
    
    # Find latest session
    base_dir = 'dataMarker'
    sessions = sorted([d for d in os.listdir(base_dir) if d.startswith('session_')])
    if not sessions:
        print("Error: No sessions found")
        return
    
    latest_session = os.path.join(base_dir, sessions[-1])
    
    # Step 2: Process
    print("\nSTEP 2: Processing")
    print("-" * 60)
    input(f"Press Enter to process {sessions[-1]}...")
    process(latest_session)
    
    print("\n" + "=" * 60)
    print("✓ Workflow completed!")
    print(f"Results saved to: {latest_session}")
    print("=" * 60)


def cleanup(session_dir=None, remove_raw=False):
    """Clean up session files."""
    if session_dir is None:
        # Find latest session
        base_dir = 'dataMarker'
        sessions = sorted([d for d in os.listdir(base_dir) if d.startswith('session_')])
        if not sessions:
            print("Error: No sessions found")
            return
        session_dir = os.path.join(base_dir, sessions[-1])
    
    print(f"Cleaning up: {session_dir}")
    
    # Remove screen captures (they take space and are not essential)
    screen_dir = os.path.join(session_dir, 'screen_captures')
    if os.path.exists(screen_dir):
        import shutil
        shutil.rmtree(screen_dir)
        print(f"✓ Removed screen_captures/")
    
    # Optionally remove raw_camera.mp4 (after processing is complete)
    if remove_raw:
        raw_video = os.path.join(session_dir, 'raw_camera.mp4')
        if os.path.exists(raw_video) and os.path.exists(os.path.join(session_dir, 'tracked.mp4')):
            os.remove(raw_video)
            print(f"✓ Removed raw_camera.mp4")
    
    print("✓ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description='Two-Phase Recording Workflow Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workflow.py record          # Start recording phase
  python workflow.py process         # Process latest session
  python workflow.py quick           # Complete workflow (record + process)
  python workflow.py list            # List all sessions
  python workflow.py cleanup -r      # Clean up latest session (remove raw video)
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Record command
    subparsers.add_parser('record', help='Start pure recording (Phase 1)')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Post-process video (Phase 2)')
    process_parser.add_argument('-s', '--session', type=str, help='Session directory to process')
    
    # Quick workflow
    subparsers.add_parser('quick', help='Run quick workflow (record + process)')
    
    # List sessions
    subparsers.add_parser('list', help='List all available sessions')
    
    # Cleanup
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up session files')
    cleanup_parser.add_argument('-s', '--session', type=str, help='Session directory to clean')
    cleanup_parser.add_argument('-r', '--remove-raw', action='store_true', help='Also remove raw_camera.mp4')
    
    args = parser.parse_args()
    
    if args.command == 'record':
        record()
    elif args.command == 'process':
        process(args.session if hasattr(args, 'session') else None)
    elif args.command == 'quick':
        quick_workflow()
    elif args.command == 'list':
        list_sessions()
    elif args.command == 'cleanup':
        session = args.session if hasattr(args, 'session') else None
        remove_raw = args.remove_raw if hasattr(args, 'remove_raw') else False
        cleanup(session, remove_raw)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
