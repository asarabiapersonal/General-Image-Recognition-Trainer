import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import streamlit as st
import os
import glob
import re
from torchinfo import summary
import seaborn as sns
from sklearn.metrics import confusion_matrix
import config
from model import Net
import torch.optim as optim
import gc

# --- CHARTING HELPERS ---
def create_confusion_matrix_chart(y_true, y_pred, class_names):
    if not y_true or not y_pred:
        return None
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 4)) 
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig.tight_layout()
        return fig
    except Exception as e:
        return None

def get_model_summary(model, input_size=(1, 3, 128, 128)):
    try:
        model_stats = summary(model, input_size=input_size, verbose=0)
        return str(model_stats)
    except Exception as e:
        return f"Error generating summary: {e}"

def get_system_metrics():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    gpu_info = "N/A"
    gpu_mem = "N/A"
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        gpu_mem = f"{allocated:.2f}GB / {reserved:.2f}GB"
    return cpu, ram, gpu_info, gpu_mem

def stop_execution():
    config.ALIVE = False    

def imshow(img, title=None):
    img = img * 0.5 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# --- FILE & MODEL MANAGEMENT ---

def get_next_model_filename():
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    pattern = os.path.join(config.MODEL_SAVE_DIR, f"{config.MODEL_NAME_BASE}_*.pth")
    existing_files = glob.glob(pattern)
    max_num = 0
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        match = re.search(r'_(\d+)\.pth$', filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    next_num = max_num + 1
    new_filename = f"{config.MODEL_NAME_BASE}_{next_num}.pth"
    return os.path.join(config.MODEL_SAVE_DIR, new_filename)

def save_checkpoint(net, optimizer, epochs, lr, batch_size, metrics):
    save_path = get_next_model_filename()
    trackers = st.session_state.get('trackers', {})
    
    plotting_history = {
        'stepTracker': list(trackers.get('stepTracker', [])),
        'batchTracker': list(trackers.get('batchTracker', [])),
        'val_step': list(trackers.get('val_step', [])),
        'val_accuracy': list(trackers.get('val_accuracy', [])),
        'val_batch_idx': int(trackers.get('val_batch_idx', 0)),
        'cm_true': list(trackers.get('cm_true', [])),
        'cm_pred': list(trackers.get('cm_pred', []))
    }

    # Retrieve custom paths from session state
    custom_test_path = st.session_state.get('custom_test_path', None)
    custom_train_path = st.session_state.get('custom_train_path', None)

    checkpoint = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': {
            'epochs': epochs,
            'learning_rate': lr,
            'batch_size': batch_size,
            'custom_test_path': custom_test_path,
            'custom_train_path': custom_train_path # Save training path
        },
        'metrics': metrics,
        'plotting_history': plotting_history
    }
    
    torch.save(checkpoint, save_path)
    
    st.session_state['last_loaded_path'] = save_path
    st.session_state['model_is_ready'] = True
    st.session_state['active_model_meta'] = checkpoint
    
    st.success(f"Model saved to {save_path}")

def load_specific_model(file_path):
    if os.path.exists(file_path):
        try:
            loaded_data = torch.load(file_path, map_location=config.DEVICE, weights_only=False)
            state_dict = None
            if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
                state_dict = loaded_data['model_state_dict']
                st.session_state['active_model_meta'] = loaded_data
                history = loaded_data.get('plotting_history', {})
                trackers = st.session_state['trackers']
                
                trackers['stepTracker'] = history.get('stepTracker', [])
                trackers['batchTracker'] = history.get('batchTracker', [])
                trackers['val_step'] = history.get('val_step', [])
                trackers['val_accuracy'] = history.get('val_accuracy', [])
                trackers['val_batch_idx'] = history.get('val_batch_idx', 0)
                trackers['cm_true'] = history.get('cm_true', [])
                trackers['cm_pred'] = history.get('cm_pred', [])

                # LOAD CUSTOM PATHS IF EXIST
                hyperparams = loaded_data.get('hyperparameters', {})
                st.session_state['custom_test_path'] = hyperparams.get('custom_test_path', None)
                st.session_state['custom_train_path'] = hyperparams.get('custom_train_path', None)

            else:
                state_dict = loaded_data
                st.session_state['active_model_meta'] = {}
                st.session_state['custom_test_path'] = None
                st.session_state['custom_train_path'] = None

            saved_num_classes = state_dict['fc3.weight'].shape[0]
            current_num_classes = st.session_state['net'].fc3.out_features
            
            if saved_num_classes != current_num_classes:
                st.warning(f"⚠️ Resizing Network: Current ({current_num_classes}) -> Saved ({saved_num_classes}) classes.")
                gc.collect()
                new_net = Net(num_classes=saved_num_classes).to(config.DEVICE)
                st.session_state['net'] = new_net
                
                saved_lr = config.LEARNING_RATE
                if 'hyperparameters' in st.session_state['active_model_meta']:
                    saved_lr = st.session_state['active_model_meta']['hyperparameters'].get('learning_rate', config.LEARNING_RATE)
                st.session_state['optimizer'] = optim.Adam(new_net.parameters(), lr=saved_lr)
            
            st.session_state['net'].load_state_dict(state_dict)
            st.session_state['last_loaded_path'] = file_path
            st.session_state['model_is_ready'] = True 
            
            st.toast(f"Loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            st.error(f"Error loading model file: {e}")
    else:
        st.error(f"File not found: {file_path}")

def display_active_model_info():
    meta = st.session_state.get('active_model_meta', {})
    if meta:
        hp = meta.get('hyperparameters', {})
        metrics = meta.get('metrics', {})
        st.sidebar.markdown("### 📜 Loaded Model Info")
        st.sidebar.info(f"**Epochs:** {hp.get('epochs', '-')}")
        st.sidebar.info(f"**Batch:** {hp.get('batch_size', '-')}")
        st.sidebar.info(f"**LR:** {hp.get('learning_rate', '-')}")
        
        # Display Custom Paths
        c_train = hp.get('custom_train_path', None)
        c_test = hp.get('custom_test_path', None)
        if c_train:
             st.sidebar.caption(f"**Train Folder:** ...{os.path.basename(c_train)}")
        if c_test:
             st.sidebar.caption(f"**Test Folder:** ...{os.path.basename(c_test)}")

        if 'final_loss' in metrics:
            st.sidebar.metric("Saved Loss", f"{metrics['final_loss']:.4f}")
        if 'val_accuracy' in metrics:
            st.sidebar.metric("Saved Accuracy", f"{metrics['val_accuracy']:.2f}%")
        st.sidebar.markdown("---")