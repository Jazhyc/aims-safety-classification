# launch_verl.py
import os
import ray
import time
import wandb

# CUDA fixes, must happen before Ray or any CUDA initialization
os.environ.pop("ROCR_VISIBLE_DEVICES", None)
os.environ.pop("HIP_VISIBLE_DEVICES", None)

from verl.trainer.main_ppo import main

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Training interrupted by error: {e}")
        raise e
    finally:
        print("\n" + "="*40)
        print("TRAINING FINISHED. STARTING CLEANUP...")
        print("="*40)
        
        # 1. Give WandB a moment to finish the final sync
        if wandb.run is not None:
            wandb.finish()
            
        # 2. Explicitly shut down Ray to kill vLLM procs gracefully
        if ray.is_initialized():
            ray.shutdown()
            
        # 3. Small sleep to allow background OS sockets to close
        time.sleep(5)
        print("Cleanup complete. Goodbye!")