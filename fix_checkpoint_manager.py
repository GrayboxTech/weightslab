"""
Script to fix corrupted checkpoint_manager_v2.py file
"""

# Read the corrupted file
with open('weightslab/components/checkpoint_manager_v2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and fix save_config method (around line 504)
# Find and fix save_data_state method

output = []
skip_until_next_def = False
in_save_config = False
in_save_data_state = False

i = 0
while i < len(lines):
    line = lines[i]

    # Detect start of save_config
    if 'def save_config(' in line and not skip_until_next_def:
        in_save_config = True
        # Write fixed save_config
        output.append('    def save_config(\n')
        output.append('        self,\n')
        output.append('        config: Dict[str, Any],\n')
        output.append('        config_name: str = "config"\n')
        output.append('    ) -> Optional[Path]:\n')
        output.append('        """Save hyperparameter configuration."""\n')
        output.append('        if self.current_exp_hash is None:\n')
        output.append('            logger.warning("No experiment hash set. Call update_experiment_hash first.")\n')
        output.append('            return None\n')
        output.append('        \n')
        output.append('        hp_hash_dir = self.hp_dir / self.current_exp_hash\n')
        output.append('        config_file = hp_hash_dir / f"{self.current_exp_hash}_{config_name}.yaml"\n')
        output.append('        \n')
        output.append('        try:\n')
        output.append('            config_with_meta = {\n')
        output.append("                'hyperparameters': config,\n")
        output.append("                'exp_hash': self.current_exp_hash,\n")
        output.append("                'last_updated': datetime.now().isoformat()\n")
        output.append('            }\n')
        output.append('            \n')
        output.append("            with open(config_file, 'w') as f:\n")
        output.append('                yaml.dump(config_with_meta, f, default_flow_style=False)\n')
        output.append('            \n')
        output.append('            logger.info(f"Saved config: {config_file.name}")\n')
        output.append('            return config_file\n')
        output.append('        except Exception as e:\n')
        output.append('            logger.error(f"Failed to save config: {e}")\n')
        output.append('            return None\n')
        output.append('    \n')
        skip_until_next_def = True
        i += 1
        continue

    # Detect start of save_data_state
    if 'def save_data_state(' in line and skip_until_next_def:
        skip_until_next_def = False
        in_save_data_state = True
        # Write fixed save_data_state
        output.append('    def save_data_state(\n')
        output.append('        self,\n')
        output.append('        data_state: Dict[str, Any],\n')
        output.append('        state_name: str = "data_state"\n')
        output.append('    ) -> Optional[Path]:\n')
        output.append('        """Save data state (UIDs, discard status, tags)."""\n')
        output.append('        if self.current_exp_hash is None:\n')
        output.append('            logger.warning("No experiment hash set. Call update_experiment_hash first.")\n')
        output.append('            return None\n')
        output.append('        \n')
        output.append('        data_hash_dir = self.data_checkpoint_dir / self.current_exp_hash\n')
        output.append('        state_file = data_hash_dir / f"{self.current_exp_hash}_{state_name}.yaml"\n')
        output.append('        \n')
        output.append('        try:\n')
        output.append('            serializable_state = {\n')
        output.append("                'uids': data_state.get('uids', []),\n")
        output.append("                'discarded': list(data_state.get('discarded', set())),\n")
        output.append("                'tags': data_state.get('tags', {}),\n")
        output.append("                'exp_hash': self.current_exp_hash,\n")
        output.append("                'last_updated': datetime.now().isoformat()\n")
        output.append('            }\n')
        output.append('            \n')
        output.append("            with open(state_file, 'w') as f:\n")
        output.append('                yaml.dump(serializable_state, f, default_flow_style=False)\n')
        output.append('            \n')
        output.append('            logger.info(f"Saved data state: {state_file.name}")\n')
        output.append('            return state_file\n')
        output.append('        except Exception as e:\n')
        output.append('            logger.error(f"Failed to save data state: {e}")\n')
        output.append('            return None\n')
        output.append('    \n')
        skip_until_next_def = True
        i += 1
        continue

    # Skip corrupted lines
    if skip_until_next_def:
        if line.strip().startswith('def ') and 'save_config' not in line and 'save_data_state' not in line:
            skip_until_next_def = False
            output.append(line)
        i += 1
        continue

    # Regular line
    output.append(line)
    i += 1

# Write fixed file
with open('weightslab/components/checkpoint_manager_v2.py', 'w', encoding='utf-8') as f:
    f.writelines(output)

print("Fixed checkpoint_manager_v2.py")
