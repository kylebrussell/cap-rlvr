#!/usr/bin/env python3
"""
Data Migration Script: Vast.ai -> Lambda Labs
Transfers processed CAP RLVR task data for RL training setup.
"""

import os
import sys
import subprocess
import hashlib
import json
import argparse
from pathlib import Path
from datetime import datetime
import tarfile
import tempfile

class DataMigrator:
    def __init__(self, vast_host="vast-cap", lambda_host=None, dry_run=False):
        self.vast_host = vast_host
        self.lambda_host = lambda_host
        self.dry_run = dry_run
        self.vast_data_path = "~/cap_rlvr/data_tasks"
        self.vast_sft_path = "~/cap_rlvr/data_tasks/sft_formatted"
        self.lambda_data_path = "~/cap_rlvr/data_tasks"
        
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def run_ssh_command(self, host, command, capture_output=True):
        """Execute SSH command on remote host"""
        full_command = ["ssh", host, command]
        if self.dry_run:
            self.log(f"DRY RUN: Would execute: {' '.join(full_command)}")
            return ""
            
        try:
            if capture_output:
                result = subprocess.run(full_command, capture_output=True, text=True, check=True)
                return result.stdout.strip()
            else:
                subprocess.run(full_command, check=True)
                return ""
        except subprocess.CalledProcessError as e:
            self.log(f"SSH command failed: {e}")
            self.log(f"stderr: {e.stderr}")
            raise
            
    def check_vast_data_ready(self):
        """Check if all data preparation tasks are completed"""
        self.log("Checking Vast.ai data preparation status...")
        
        # Check for running prep processes
        running_procs = self.run_ssh_command(
            self.vast_host, 
            "ps aux | grep 'python prep_' | grep -v grep | wc -l"
        )
        
        if int(running_procs) > 0:
            self.log("⚠️  Data prep processes still running on Vast.ai")
            self.run_ssh_command(self.vast_host, "ps aux | grep 'python prep_' | grep -v grep", False)
            return False
            
        # Check for expected output directories
        expected_tasks = ["bluebook", "holding", "entail", "summarise", "retrieval"]
        missing_tasks = []
        
        for task in expected_tasks:
            try:
                self.run_ssh_command(
                    self.vast_host,
                    f"ls {self.vast_data_path}/{task}/train.jsonl > /dev/null 2>&1"
                )
            except subprocess.CalledProcessError:
                missing_tasks.append(task)
                
        if missing_tasks:
            self.log(f"⚠️  Missing tasks: {', '.join(missing_tasks)}")
            return False
            
        self.log("✅ All data preparation tasks completed on Vast.ai")
        return True
        
    def get_data_inventory(self):
        """Get inventory of data files and sizes"""
        self.log("Creating data inventory...")
        
        inventory = {'raw_tasks': {}, 'sft_formatted': {}}
        
        # Inventory raw task data
        tasks_info = self.run_ssh_command(
            self.vast_host,
            f"cd {self.vast_data_path} && find . -name '*.jsonl' -exec ls -lh {{}} \\; 2>/dev/null || echo 'No raw task files found'"
        )
        
        for line in tasks_info.split('\n'):
            if line.strip() and not line.startswith('No raw'):
                parts = line.split()
                if len(parts) >= 9:
                    size = parts[4]
                    filepath = parts[8]
                    task_name = filepath.split('/')[1] if '/' in filepath else 'unknown'
                    
                    if task_name not in inventory['raw_tasks']:
                        inventory['raw_tasks'][task_name] = []
                    inventory['raw_tasks'][task_name].append({'file': filepath, 'size': size})
        
        # Get total sizes for raw tasks
        for task in inventory['raw_tasks'].keys():
            total_size = self.run_ssh_command(
                self.vast_host,
                f"cd {self.vast_data_path} && du -sh {task} 2>/dev/null | cut -f1 || echo '0'"
            )
            inventory['raw_tasks'][task].append({'total_size': total_size})
        
        # Inventory SFT formatted data (if exists)
        sft_info = self.run_ssh_command(
            self.vast_host,
            f"cd {self.vast_sft_path} && find . -name '*.jsonl' -exec ls -lh {{}} \\; 2>/dev/null || echo 'No SFT formatted files found'"
        )
        
        if "No SFT formatted files found" not in sft_info:
            for line in sft_info.split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 9:
                        size = parts[4]
                        filepath = parts[8]
                        format_type = filepath.split('/')[1] if '/' in filepath else 'unknown'
                        
                        if format_type not in inventory['sft_formatted']:
                            inventory['sft_formatted'][format_type] = []
                        inventory['sft_formatted'][format_type].append({'file': filepath, 'size': size})
            
            # Get total size of SFT formatted data
            sft_total_size = self.run_ssh_command(
                self.vast_host,
                f"du -sh {self.vast_sft_path} 2>/dev/null | cut -f1 || echo '0'"
            )
            inventory['sft_formatted']['_total_size'] = sft_total_size
        
        return inventory
        
    def create_compressed_archive(self, output_path="/tmp/cap_rlvr_data.tar.gz"):
        """Create compressed archive of all task data including SFT formatted data"""
        self.log("Creating compressed archive on Vast.ai...")
        
        # Build archive command - include both raw and SFT formatted data
        archive_cmd = (
            f"cd ~/cap_rlvr && "
            f"tar -czf {output_path} "
            f"--exclude='*.log' --exclude='__pycache__' "
            f"data_tasks/bluebook/ data_tasks/holding/ data_tasks/entail/ data_tasks/summarise/ "
            f"data_tasks/retrieval/ data_tasks/sft_formatted/ 2>/dev/null || "
            f"tar -czf {output_path} "
            f"--exclude='*.log' --exclude='__pycache__' "
            f"data_tasks/bluebook/ data_tasks/holding/ data_tasks/entail/ data_tasks/summarise/ "
            f"data_tasks/sft_formatted/ 2>/dev/null || "
            f"tar -czf {output_path} "
            f"--exclude='*.log' --exclude='__pycache__' "
            f"data_tasks/bluebook/ data_tasks/holding/ data_tasks/entail/ data_tasks/summarise/"
        )
        
        self.run_ssh_command(self.vast_host, archive_cmd, False)
        
        # Get archive size and checksum
        archive_size = self.run_ssh_command(self.vast_host, f"ls -lh {output_path} | awk '{{print $5}}'")
        archive_checksum = self.run_ssh_command(self.vast_host, f"md5sum {output_path} | awk '{{print $1}}'")
        
        self.log(f"✅ Archive created: {archive_size}, MD5: {archive_checksum[:8]}...")
        return output_path, archive_size, archive_checksum
        
    def transfer_to_lambda(self, archive_path, checksum):
        """Transfer compressed archive to Lambda Labs"""
        if not self.lambda_host:
            self.log("⚠️  Lambda host not specified, skipping transfer")
            self.log(f"Archive ready for manual transfer: {archive_path}")
            self.log(f"Expected MD5: {checksum}")
            return False
            
        self.log(f"Transferring archive to Lambda Labs ({self.lambda_host})...")
        
        # Create destination directory
        self.run_ssh_command(self.lambda_host, "mkdir -p ~/cap_rlvr", False)
        
        # Transfer archive
        transfer_cmd = [
            "scp", 
            f"{self.vast_host}:{archive_path}",
            f"{self.lambda_host}:~/cap_rlvr_data.tar.gz"
        ]
        
        if not self.dry_run:
            subprocess.run(transfer_cmd, check=True)
        else:
            self.log(f"DRY RUN: Would execute: {' '.join(transfer_cmd)}")
            
        # Verify checksum on Lambda
        lambda_checksum = self.run_ssh_command(
            self.lambda_host, 
            "md5sum ~/cap_rlvr_data.tar.gz | awk '{print $1}'"
        )
        
        if lambda_checksum != checksum:
            raise Exception(f"Checksum mismatch! Expected: {checksum}, Got: {lambda_checksum}")
            
        self.log("✅ Transfer completed and verified")
        return True
        
    def extract_on_lambda(self):
        """Extract archive on Lambda Labs"""
        if not self.lambda_host:
            return False
            
        self.log("Extracting archive on Lambda Labs...")
        
        extract_commands = [
            "mkdir -p ~/cap_rlvr",
            "cd ~/cap_rlvr && tar -xzf ~/cap_rlvr_data.tar.gz",
            "ls -la ~/cap_rlvr/data_tasks/",
            "du -sh ~/cap_rlvr/data_tasks/*"
        ]
        
        for cmd in extract_commands:
            self.run_ssh_command(self.lambda_host, cmd, False)
            
        self.log("✅ Extraction completed on Lambda Labs")
        return True
        
    def cleanup_temp_files(self, archive_path):
        """Clean up temporary files"""
        self.log("Cleaning up temporary files...")
        
        # Remove archive from Vast.ai
        self.run_ssh_command(self.vast_host, f"rm -f {archive_path}", False)
        
        if self.lambda_host:
            # Remove compressed archive from Lambda (keep extracted data)
            self.run_ssh_command(self.lambda_host, "rm -f ~/cap_rlvr_data.tar.gz", False)
            
        self.log("✅ Cleanup completed")
        
    def run_migration(self):
        """Execute full migration process"""
        self.log("🚀 Starting CAP RLVR data migration: Vast.ai -> Lambda Labs")
        
        try:
            # 1. Check data readiness
            if not self.check_vast_data_ready():
                self.log("❌ Data not ready for migration")
                return False
                
            # 2. Create inventory
            inventory = self.get_data_inventory()
            self.log("📊 Data inventory:")
            
            # Display raw tasks
            self.log("  Raw Tasks:")
            for task, files in inventory['raw_tasks'].items():
                total_size = files[-1].get('total_size', 'unknown') if files else 'unknown'
                self.log(f"    {task}: {total_size}")
            
            # Display SFT formatted data (if exists)
            if inventory['sft_formatted'] and len(inventory['sft_formatted']) > 0:
                sft_total = inventory['sft_formatted'].get('_total_size', 'unknown')
                self.log(f"  SFT Formatted: {sft_total}")
                for format_type, files in inventory['sft_formatted'].items():
                    if format_type != '_total_size':
                        self.log(f"    {format_type}: {len(files)} files")
            else:
                self.log("  SFT Formatted: Not found (run format_for_sft.py first)")
                
            # 3. Create compressed archive
            archive_path, archive_size, checksum = self.create_compressed_archive()
            
            # 4. Transfer to Lambda
            if self.transfer_to_lambda(archive_path, checksum):
                # 5. Extract on Lambda
                self.extract_on_lambda()
                
            # 6. Cleanup
            self.cleanup_temp_files(archive_path)
            
            self.log("🎉 Migration completed successfully!")
            self.log("Next steps:")
            self.log("  1. Verify data integrity on Lambda Labs")
            self.log("  2. Check SFT formatted data in data_tasks/sft_formatted/")
            self.log("  3. Set up RL training environment") 
            self.log("  4. Run build_faiss.py for retrieval embeddings if needed")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Migration failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Migrate CAP RLVR data from Vast.ai to Lambda Labs")
    parser.add_argument("--vast-host", default="vast-cap", help="Vast.ai SSH host alias")
    parser.add_argument("--lambda-host", help="Lambda Labs SSH host (required for transfer)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    parser.add_argument("--check-only", action="store_true", help="Only check if data is ready")
    
    args = parser.parse_args()
    
    migrator = DataMigrator(
        vast_host=args.vast_host,
        lambda_host=args.lambda_host, 
        dry_run=args.dry_run
    )
    
    if args.check_only:
        ready = migrator.check_vast_data_ready()
        inventory = migrator.get_data_inventory()
        print("\n📊 Current data status:")
        
        # Display raw tasks
        print("  Raw Tasks:")
        for task, files in inventory.get('raw_tasks', {}).items():
            total_size = files[-1].get('total_size', 'unknown') if files else 'unknown'
            print(f"    {task}: {total_size}")
        
        # Display SFT formatted data (if exists)
        sft_data = inventory.get('sft_formatted', {})
        if sft_data and len(sft_data) > 0:
            sft_total = sft_data.get('_total_size', 'unknown')
            print(f"  SFT Formatted: {sft_total}")
            for format_type, files in sft_data.items():
                if format_type != '_total_size':
                    print(f"    {format_type}: {len(files)} files")
        else:
            print("  SFT Formatted: Not found (run format_for_sft.py first)")
            
        sys.exit(0 if ready else 1)
        
    if not args.lambda_host and not args.dry_run:
        print("⚠️  Warning: No Lambda host specified. Archive will be created but not transferred.")
        print("Use --lambda-host to specify destination, or --dry-run to test")
        
    success = migrator.run_migration()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()